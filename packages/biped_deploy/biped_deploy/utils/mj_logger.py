import json
from dataclasses import asdict, dataclass

import mujoco
import numpy as np


# Save safety checker data
def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects"""
    if isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()
    err_msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(err_msg)


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

    action_rate: dict[str, float]
    joint_pos_rate: dict[str, float]


@dataclass
class Limits:
    joint_pos_limits: dict[str, float]

    total_mass_force: float



class MJLogger:
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.metrics_data: list[Metrics] = []

        self.prev_joint_pos = {}
        self.prev_action = {}

    def record_limits(self):
        joint_pos_limits = {}

        for jnt_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jnt_id,
            )
            joint_pos_limits[joint_name] = self.model.jnt_range[jnt_id]

        total_mass_force = np.sum(self.model.body_mass) * np.linalg.norm(self.model.opt.gravity)

        limits = Limits(
            joint_pos_limits=joint_pos_limits,
            total_mass_force=total_mass_force,
        )

        self.metrics_data.append(limits)

    def record_metrics(self, current_time):
        # Create joint position and velocity dictionaries
        joint_pos = {}
        joint_vel = {}
        applied_torques = {}

        joint_pos_rate = {}
        action_rate = {}

        # Loop through joints to collect data
        for jnt_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jnt_id,
            )

            if "floating_base" in joint_name:
                continue

            # Position
            joint_pos_addr = self.model.jnt_qposadr[jnt_id]
            qpos = self.data.qpos[joint_pos_addr]
            joint_pos[joint_name] = qpos

            # Joint Pos rate
            joint_pos_rate[joint_name] = qpos - self.prev_joint_pos.get(joint_name, qpos)
            self.prev_joint_pos[joint_name] = qpos

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

                        # Action rate
                        action_rate[joint_name] = joint_torque - self.prev_action.get(joint_name, joint_torque)
                        self.prev_action[joint_name] = joint_torque

        # Get foot contact forces
        foot_contact_forces = self._get_foot_contact_forces()

        # Create metrics object
        metrics = Metrics(
            timestamp=current_time,
            base_lin_pos=self.data.qpos[:3].copy(),
            base_quat_pos=self.data.qpos[3:7].copy(),
            joint_pos=joint_pos,
            base_lin_vel=self.data.qvel[:3].copy(),
            base_quat_vel=self.data.qvel[3:6].copy(),
            joint_vel=joint_vel,
            applied_torques=applied_torques,
            foot_contact_forces=foot_contact_forces,
            action_rate=action_rate,
            joint_pos_rate=joint_pos_rate,
        )

        # Store metrics
        self.metrics_data.append(metrics)

    def save_data(self, log_dir):
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

        foot_forces = dict.fromkeys(foot_bodies, 0.0)

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
