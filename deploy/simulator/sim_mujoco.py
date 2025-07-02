import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np


class ElasticBand:
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0.0
        self.enable = True

    def advance(self, x, v):
        """
        Args:
          dx: desired position - current position
          v: current velocity
        """
        dx = self.point - x
        distance = np.linalg.norm(dx)
        direction = dx / distance
        v = np.dot(v, direction)
        return (self.stiffness * (distance - self.length) - self.damping * v) * direction


@dataclass
class Metrics:
    timestamp: float

    base_lin_pos: np.ndarray
    base_quat_pos: np.ndarray
    joint_pos: dict[str, float]

    base_lin_vel: np.ndarray
    base_quat_vel: np.ndarray
    joint_vel: dict[str, float]

    applied_torques: dict[str, float]
    foot_contact_forces: dict[str, float]


@dataclass
class SafetyViolation:
    timestamp: float
    joint_name: str
    check_type: str
    value: float
    limit: float
    additional_info: dict[str, Any] = field(default_factory=dict)


# Save safety checker data
def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects"""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    err_msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(err_msg)


class MujocoSim:
    def __init__(self, scene_path, config):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = config["sim_dt"]
        self.current_time = 0
        self.episode_length = config["episode_length"]
        self.data = mujoco.MjData(self.model)

        self.real_time = config["real_time"]
        self.render_dt = config["render_dt"]

        self.sim_lock = threading.Lock()

        self.check_violations = config["check_violations"]
        self.safety_checker_verbose = config["safety_checker_verbose"]
        self.safety_violations: list[SafetyViolation] = []
        self.metrics_data: list[Metrics] = []

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Enable the weld constraint
        self.model.eq_active0[0] = 1 if config["fix_base"] else 0

        self.reset()

        self.enable_keyboard = config["enable_keyboard"]
        self.keyboard_lock = threading.Lock()
        self.controller_command = np.zeros(3)

        self.enable_GUI = config["enable_GUI"]
        if self.enable_GUI:
            self.close_event = threading.Event()
            self.viewer_thread = threading.Thread(target=self.run_render, args=(self.close_event,))
            self.viewer_thread.start()

        self.elastic_band_enabled = config["elastic_band"]
        if self.elastic_band_enabled:
            self.elastic_band = ElasticBand()
            self.band_attached_link = self.model.body("torso_link").id

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def sim_step(self, torques):
        step_start = time.perf_counter()
        self._apply_torques(torques)
        if self.check_violations:
            self._safety_check()
            self._metrics()

        with self.sim_lock:
            mujoco.mj_step(self.model, self.data)

        if self.elastic_band_enabled:
            self.data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.advance(
                self.data.qpos[:3],
                self.data.qvel[:3],
            )

        if self.real_time:
            time_to_wait = max(0, step_start - time.perf_counter() + self.model.opt.timestep)
            time.sleep(time_to_wait)

        self.current_time += self.model.opt.timestep

    def get_controller_command(self):
        with self.keyboard_lock:
            return self.controller_command

    def close(self, log_dir=None):
        # Close Mujoco viewer if opened
        if self.enable_GUI:
            self.close_event.set()
            self.viewer_thread.join()

        if self.check_violations and log_dir is not None:
            self._save_violations(log_dir)

    def _record_violation(
        self,
        joint_name: str,
        check_type: str,
        value: float,
        limit: float,
        additional_info: dict[str, Any] = None,
    ):
        violation = SafetyViolation(
            timestamp=self.current_time,
            joint_name=joint_name,
            check_type=check_type,
            value=value,
            limit=limit,
            additional_info=additional_info or {},
        )
        self.safety_violations.append(violation)

        if self.safety_checker_verbose:
            print(
                f"[{violation.timestamp:.3f}s] {joint_name}: {check_type.upper()} violation - value={value:.4f}, limit={limit}",
            )

    def _safety_check(self):
        # Loop through joints
        for jnt_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)

            # Position check
            joint_pos_addr = self.model.jnt_qposadr[jnt_id]
            joint_value = self.data.qpos[joint_pos_addr]

            # Velocity check
            joint_vel_addr = self.model.jnt_dofadr[jnt_id]
            joint_velocity = self.data.qvel[joint_vel_addr] if joint_vel_addr >= 0 else 0

            # Torque check
            joint_torque = 0.0
            for act_id in range(self.model.nu):
                if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                    trn_joint_id = self.model.actuator_trnid[act_id, 0]
                    if trn_joint_id == jnt_id:
                        joint_torque += self.data.actuator_force[act_id]

            # Check position limits
            if self.model.jnt_limited[jnt_id]:
                joint_limits = self.model.jnt_range[jnt_id]
                pos_in_range = joint_limits[0] <= joint_value <= joint_limits[1]
                if not pos_in_range:
                    self._record_violation(
                        joint_name=joint_name,
                        check_type="position",
                        value=joint_value,
                        limit=joint_limits,
                        additional_info={
                            "lower_limit": joint_limits[0],
                            "upper_limit": joint_limits[1],
                        },
                    )

            # Check velocity limits
            if hasattr(self.model, "jnt_vel_limits") and jnt_id < len(self.model.jnt_vel_limits):
                vel_limit = self.model.jnt_vel_limits[jnt_id]
                vel_in_range = abs(joint_velocity) <= vel_limit
                if not vel_in_range:
                    self._record_violation(
                        joint_name=joint_name,
                        check_type="velocity",
                        value=joint_velocity,
                        limit=vel_limit,
                    )

            # Check torque limits
            if hasattr(self.model, "jnt_torque_limits") and jnt_id < len(self.model.jnt_torque_limits):
                torque_limit = self.model.jnt_torque_limits[jnt_id]
                torque_in_range = abs(joint_torque) <= torque_limit
                if not torque_in_range:
                    self._record_violation(
                        joint_name=joint_name,
                        check_type="torque",
                        value=joint_torque,
                        limit=torque_limit,
                    )

        # Contact force checks
        foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_bodies}

        foot_forces = {name: [] for name in foot_bodies}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]

            force_vec = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force_vec)
            force = force_vec[:3]
            force_norm = np.linalg.norm(force)

            for foot_name, foot_id in foot_ids.items():
                if foot_id in (geom1_body, geom2_body):
                    foot_forces[foot_name].append(force_norm)

        # Record contact force violations
        total_mass_force = np.sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity)
        for foot, forces in foot_forces.items():
            total = sum(forces)
            if total > total_mass_force:
                self._record_violation(
                    joint_name=foot,
                    check_type="contact_force",
                    value=total,
                    limit=total_mass_force,
                    additional_info={
                        "individual_forces": forces,
                        "num_contacts": len(forces),
                    },
                )

    def _metrics(self):
        # Create joint position and velocity dictionaries
        joint_pos = {}
        joint_vel = {}
        applied_torques = {}

        # Loop through joints to collect data
        for jnt_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jnt_id,
            )

            # Position
            joint_pos_addr = self.model.jnt_qposadr[jnt_id]
            joint_pos[joint_name] = self.data.qpos[joint_pos_addr]

            # Velocity
            joint_vel_addr = self.model.jnt_dofadr[jnt_id]
            joint_vel[joint_name] = self.data.qvel[joint_vel_addr] if joint_vel_addr >= 0 else 0

            # Torque
            joint_torque = 0.0
            for act_id in range(self.model.nu):
                if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                    trn_joint_id = self.model.actuator_trnid[act_id, 0]
                    if trn_joint_id == jnt_id:
                        joint_torque = self.data.actuator_force[act_id]
                        applied_torques[joint_name] = joint_torque

        # Get foot contact forces
        foot_contact_forces = self._get_foot_contact_forces()

        # Create metrics object
        metrics = Metrics(
            timestamp=self.current_time,
            base_lin_pos=self.data.qpos[:3].copy(),
            base_quat_pos=self.data.qpos[3:7].copy(),
            joint_pos=joint_pos,
            base_lin_vel=self.data.qvel[:3].copy(),
            base_quat_vel=self.data.qvel[3:6].copy(),
            joint_vel=joint_vel,
            applied_torques=applied_torques,
            foot_contact_forces=foot_contact_forces,
        )

        # Store metrics
        self.metrics_data.append(metrics)

    def _save_violations(self, log_dir):
        safety_checker_path = log_dir / "safety_check.json"
        with safety_checker_path.open("w") as f:
            json.dump(
                [asdict(v) for v in self.safety_violations],
                f,
                indent=2,
                default=_json_serializer,
            )
        print(f"Saved violations to {safety_checker_path}")

        # Save metrics data
        metrics_path = log_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(
                [asdict(m) for m in self.metrics_data],
                f,
                indent=2,
                default=_json_serializer,
            )
        print(f"Saved metrics to {metrics_path}")

    def _get_foot_contact_forces(self) -> dict[str, float]:
        """Calculate contact forces for each foot"""
        foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_bodies}

        foot_forces = {name: 0.0 for name in foot_bodies}

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]

            force_vec = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force_vec)
            force_norm = np.linalg.norm(force_vec[:3])

            for foot_name, foot_id in foot_ids.items():
                if foot_id in (geom1_body, geom2_body):
                    foot_forces[foot_name] += force_norm

        return foot_forces

    def _apply_torques(self, torques):
        self.data.ctrl[:] = torques

    def get_robot_state(self):
        with self.sim_lock:
            return {
                "base_orientation": self.data.qpos[3:7],
                "qpos": self.data.qpos[7:],
                "base_angular_vel": self.data.qvel[3:6],
                "qvel": self.data.qvel[6:],
            }

    def run_render(self, close_event):
        key_cb = self.key_callback if self.enable_keyboard else None
        viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb)
        while viewer.is_running() and not close_event.is_set():
            with self.sim_lock:
                viewer.sync()
            time.sleep(self.render_dt)
        viewer.close()

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        with self.keyboard_lock:
            if key == glfw.KEY_UP or key == glfw.KEY_KP_8:
                self.controller_command[0] += 0.1
            elif key == glfw.KEY_DOWN or key == glfw.KEY_KP_5:
                self.controller_command[0] -= 0.1
            elif key == glfw.KEY_LEFT or key == glfw.KEY_KP_4:
                self.controller_command[1] += 0.1
            elif key == glfw.KEY_RIGHT or key == glfw.KEY_KP_6:
                self.controller_command[1] -= 0.1
            elif key == glfw.KEY_Z or key == glfw.KEY_KP_7:
                self.controller_command[2] += 0.1
            elif key == glfw.KEY_X or key == glfw.KEY_KP_9:
                self.controller_command[2] -= 0.1
            elif key == glfw.KEY_B:
                self.elastic_band_enabled = not self.elastic_band_enabled
            elif key == glfw.KEY_I:
                self.elastic_band.length += 0.1
            elif key == glfw.KEY_K:
                self.elastic_band.length -= 0.1
