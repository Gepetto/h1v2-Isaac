import json
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

# Save safety checker data
def _json_serializer(obj):
    """Handle numpy types and other non-serializable objects"""
    if isinstance(obj, np.ndarray | np.generic):
        return obj.tolist()
    err_msg = f"Object of type {type(obj)} is not JSON serializable"
    raise TypeError(err_msg)

@dataclass
class SafetyViolation:
    timestamp: float
    joint_name: str
    check_type: str
    value: float
    limit: float
    additional_info: dict[str, Any] = field(default_factory=dict)

class SafetyChecker:
    def __init__(self):
        self.safety_violations: list[SafetyViolation] = []

    def check_safety(self, json_file):
        # Load data from file
        with open(json_file) as file:
            data = json.load(file)

            self._extract_limits(data)
            for entry in data[1:]:
                current_time = entry["timestamp"]
                self._check_joint_position(current_time, entry["joint_pos"])
                self._check_contact_forces(current_time, entry["foot_contact_forces"])

    def _extract_limits(self, data):
        entry = data[0]

        self.joint_pos_limits = entry["joint_pos_limits"]
        self.joint_vel_limits = entry.get("joint_vel_limits", None)
        self.joint_torque_limits = entry.get("joint_torque_limits", None)
        self.total_mass_force = entry["total_mass_force"]

    def _check_joint_position(self, current_time, joint_pos):
        for joint, position in joint_pos.items():
            limits = self.joint_pos_limits[joint]
            pos_in_range = limits[0] <= position <= limits[1]
            if not pos_in_range:
                self._record_violation(
                    current_time=current_time,
                    joint_name=joint,
                    check_type="position",
                    value=position,
                    limit=limits,
                    additional_info={
                        "lower_limit": limits[0],
                        "upper_limit": limits[1],
                    },
                )

    def _check_joint_vel(self, current_time, joint_vel):
        for joint, velocity in joint_vel.items():
            vel_limit = self.joint_vel_limits[joint]
            vel_in_range = abs(velocity) <= vel_limit
            if not vel_in_range:
                self._record_violation(
                    current_time=current_time,
                    joint_name=joint,
                    check_type="velocity",
                    value=velocity,
                    limit=vel_limit,
                )

    def _check_torque(self, current_time, applied_torques):
        for joint, torque in applied_torques.items():
            torque_limit = self.joint_torque_limits[joint]
            torque_in_range = abs(torque) <= torque_limit
            if not torque_in_range:
                self._record_violation(
                    current_time=current_time,
                    joint_name=joint,
                    check_type="torque",
                    value=torque,
                    limit=torque_limit,
                )

    def _check_contact_forces(self, current_time, foot_contact_forces):
        # Record contact force violations
        for foot, force in foot_contact_forces.items():
            if force > self.total_mass_force:
                self._record_violation(
                    current_time=current_time,
                    joint_name=foot,
                    check_type="contact_force",
                    value=force,
                    limit=self.total_mass_force,
                )

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

    def _record_violation(
        self,
        current_time: int,
        joint_name: str,
        check_type: str,
        value: float,
        limit: float,
        additional_info: dict[str, Any] | None = None,
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
