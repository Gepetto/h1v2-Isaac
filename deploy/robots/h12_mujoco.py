import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from biped_assets import SCENE_PATHS


@dataclass
class SafetyViolation:
    timestamp: float
    joint_name: str
    check_type: str
    value: float
    limit: float
    additional_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        data = asdict(self)
        for key, value in data["additional_info"].items():
            if isinstance(value, np.ndarray):
                data["additional_info"][key] = value.tolist()
        return data


class H1Mujoco:
    def __init__(
        self,
        scene_path,
        config,
    ):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = config["dt"]
        self.current_time = 0
        self.episode_length = config["episode_length"]
        self.data = mujoco.MjData(self.model)
        self.real_time = config["real_time"]

        self.enable_keyboard = config["enable_keyboard"]
        self.controller_command = np.zeros(3)
        self.keyboard_lock = threading.Lock()

        self.sim_lock = threading.Lock()

        self.safety_violations: list[SafetyViolation] = []

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.kp = np.array(config["kp"])
        self.kd = np.array(config["kd"])
        self.decimation = config["decimation"]

        # Enable the weld constraint
        if config["fix_base"]:
            self.model.eq_active0[0] = 1
        else:
            self.model.eq_active0[0] = 0

        self.reset()

        if config["enable_GUI"]:
            self.close_event = threading.Event()
            self.thread = threading.Thread(target=self.run_render, args=(self.close_event,))
            self.thread.start()

        self.safety_checker_verbose = config["safety_checker_verbose"]

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def step(self, q_ref):
        for _ in range(self.decimation):
            step_start = time.perf_counter()
            torques = self._pd_control(q_ref)
            self._apply_torques(torques)
            self._safety_check()
            with self.sim_lock:
                mujoco.mj_step(self.model, self.data)

            if self.real_time:
                time_to_wait = max(0, step_start - time.perf_counter() + self.model.opt.timestep)
                time.sleep(time_to_wait)

            self.current_time += self.model.opt.timestep

    def close(self, log_dir):
        # Close Mujoco viewer if opened
        if hasattr(self, "thread"):
            self.close_event.set()

        # Save safety checker datas
        def _json_serializer(obj):
            """Handle numpy types and other non-serializable objects"""
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        safety_checker_path = log_dir / "safety_check.json"
        with safety_checker_path.open("w") as f:
            json.dump(
                [asdict(v) for v in self.safety_violations],
                f,
                indent=2,
                default=_json_serializer,
            )
        print(f"Saved violations to {safety_checker_path}")

    def get_controller_command(self):
        with self.keyboard_lock:
            return self.controller_command

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
                f"[{violation.timestamp:.3f}s] {joint_name}: {check_type.upper()} violation - "
                f"value={value:.4f}, limit={limit:.4f}",
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

    def _apply_torques(self, torques):
        self.data.ctrl[:] = torques

    def _pd_control(self, q_ref):
        q_error = q_ref - self.data.qpos[7:]
        q_dot_error = np.zeros_like(q_ref) - self.data.qvel[6:]
        return self.kp * q_error + self.kd * q_dot_error

    def get_robot_state(self):
        return {
            "base_orientation": self.data.qpos[3:7],
            "q_pos": self.data.qpos[7:],
            "base_angular_vel": self.data.qvel[3:6],
            "q_vel": self.data.qvel[6:],
        }

    def run_render(self, close_event):
        key_cb = self.key_callback if self.enable_keyboard else None
        viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb)
        while viewer.is_running() and not close_event.is_set():
            with self.sim_lock:
                viewer.sync()
            time.sleep(0.02)  # 50 Hz
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


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]
    sim = H1Mujoco(scene_path, config["mujoco"])

    state = sim.get_robot_state()
    while True:
        sim.step(state["q_pos"])
