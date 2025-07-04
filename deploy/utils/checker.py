import json
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np


# Save safety checker data
def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects"""
    if isinstance(obj, (np.ndarray, np.generic)):
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
class SafetyViolation:
    timestamp: float
    joint_name: str
    check_type: str
    value: float
    limit: float
    additional_info: dict[str, Any] = field(default_factory=dict)


class Checker:
    def __init__(self, model, data, verbose):
        self.model = model
        self.data = data

        self.safety_checker_verbose = verbose
        self.safety_violations: list[SafetyViolation] = []
        self.metrics_data: list[Metrics] = []
        self.metrics_data: list[Metrics] = []

        self.prev_joint_pos = {}
        self.prev_action = {}

    def _record_violation(
        self,
        current_time: int,
        joint_name: str,
        check_type: str,
        value: float,
        limit: float,
        additional_info: dict[str, Any] = None,
    ):
        violation = SafetyViolation(
            timestamp=current_time,
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

    def check_safety(self, current_time):
        # Loop through joints
        for jnt_id in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id
            )

            # Position check
            joint_pos_addr = self.model.jnt_qposadr[jnt_id]
            joint_value = self.data.qpos[joint_pos_addr]

            # Velocity check
            joint_vel_addr = self.model.jnt_dofadr[jnt_id]
            joint_velocity = (
                self.data.qvel[joint_vel_addr] if joint_vel_addr >= 0 else 0
            )

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
                        current_time=current_time,
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
            if hasattr(self.model, "jnt_vel_limits") and jnt_id < len(
                self.model.jnt_vel_limits
            ):
                vel_limit = self.model.jnt_vel_limits[jnt_id]
                vel_in_range = abs(joint_velocity) <= vel_limit
                if not vel_in_range:
                    self._record_violation(
                        current_time=current_time,
                        joint_name=joint_name,
                        check_type="velocity",
                        value=joint_velocity,
                        limit=vel_limit,
                    )

            # Check torque limits
            if hasattr(self.model, "jnt_torque_limits") and jnt_id < len(
                self.model.jnt_torque_limits
            ):
                torque_limit = self.model.jnt_torque_limits[jnt_id]
                torque_in_range = abs(joint_torque) <= torque_limit
                if not torque_in_range:
                    self._record_violation(
                        current_time=current_time,
                        joint_name=joint_name,
                        check_type="torque",
                        value=joint_torque,
                        limit=torque_limit,
                    )

        # Contact force checks
        foot_bodies = ["left_ankle_roll_link", "right_ankle_roll_link"]
        foot_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in foot_bodies
        }

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
        total_mass_force = np.sum(self.model.body_mass) * np.linalg.norm(
            self.model.opt.gravity
        )
        for foot, forces in foot_forces.items():
            total = sum(forces)
            if total > total_mass_force:
                self._record_violation(
                    current_time=current_time,
                    joint_name=foot,
                    check_type="contact_force",
                    value=total,
                    limit=total_mass_force,
                    additional_info={
                        "individual_forces": forces,
                        "num_contacts": len(forces),
                    },
                )

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

            # Position
            joint_pos_addr = self.model.jnt_qposadr[jnt_id]
            qpos = self.data.qpos[joint_pos_addr]
            joint_pos[joint_name] = qpos

            joint_pos_rate[joint_name] = qpos - self.prev_joint_pos.get(joint_name,qpos)

            self.prev_joint_pos[joint_name] = qpos

            # Velocity
            joint_vel_addr = self.model.jnt_dofadr[jnt_id]
            joint_vel[joint_name] = (
                self.data.qvel[joint_vel_addr] if joint_vel_addr >= 0 else 0
            )

            # Torque
            joint_torque = 0.0
            for act_id in range(self.model.nu):
                if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                    trn_joint_id = self.model.actuator_trnid[act_id, 0]
                    if trn_joint_id == jnt_id:
                        joint_torque = self.data.actuator_force[act_id]
                        applied_torques[joint_name] = joint_torque

                        action_rate[joint_name] = joint_torque - self.prev_action.get(joint_name,joint_torque)
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
        # Save violations
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
        foot_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in foot_bodies
        }

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
