import contextlib
import numpy as np
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from robot_deploy.input_devices import Button, InputDevice
from robot_deploy.robots.robot import Robot
from robot_deploy.utils.rotation import transform_imu_data


class ConfigError(Exception): ...


class H12Real(Robot):
    REAL_JOINT_NAME_ORDER = (
        "left_hip_yaw_joint",
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    def __init__(self, config: dict, input_device: InputDevice | None = None) -> None:
        super().__init__(config, input_device)
        self.input_device = input_device

        with contextlib.suppress(Exception):
            ChannelFactoryInitialize(0, config["real"]["net_interface"])

        self.step_time = time.perf_counter()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = 0  # MotorMode.PR in unitree code
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.num_joints_total = len(self.low_cmd.motor_cmd)  # type: ignore

        # Wait for the subscriber to receive data
        self.wait_for_low_state()

        self.init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def get_joint_names(self) -> list[str]:
        return list(self.REAL_JOINT_NAME_ORDER)

    def initialize(self) -> None:
        if self.input_device is not None:
            self.input_device.wait_for(Button.start)

        print("Moving to default pos.")

        # fmt: off
        default_joint_pos = np.array([
            0.0, -0.16, 0.0, 0.36, -0.2, 0.0,   # Left lower body
            0.0, -0.16, 0.0, 0.36, -0.2, 0.0,   # Right lower body
            0.0,                                # Torso
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left upper body
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right upper body
        ])
        init_kps = np.array([
            200.0, 200.0, 200.0, 300.0, 40.0, 40.0,
            200.0, 200.0, 200.0, 300.0, 40.0, 40.0,
            300.0,
            120.0, 120.0, 120.0, 80.0, 80.0, 80.0, 80.0,
            120.0, 120.0, 120.0, 80.0, 80.0, 80.0, 80.0,
        ])
        init_kds = np.array([
            2.5, 2.5, 2.5, 4.0, 2.0, 2.0,
            2.5, 2.5, 2.5, 4.0, 2.0, 2.0,
            3.0,
            2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
        ])
        # fmt: on

        # First, set legs and raise shoulders to avoid hitting itself
        shoulder_idx = [
            self.REAL_JOINT_NAME_ORDER.index("left_shoulder_roll_joint"),
            self.REAL_JOINT_NAME_ORDER.index("right_shoulder_roll_joint"),
        ]
        t_pose = default_joint_pos.copy()
        t_pose[shoulder_idx[0]] = 0.6
        t_pose[shoulder_idx[1]] = -0.6
        self.move_to_pos(range(27), t_pose, init_kps, init_kds, 0.02, 2)

        # Then set up the default position
        self.move_to_pos(shoulder_idx, default_joint_pos[shoulder_idx], init_kps, init_kds, 0.02, 1.5)

        print("Reached default pos state.")
        if self.input_device is not None:
            self.input_device.wait_for(Button.A)

    def get_robot_state(self):
        base_orientation, base_angular_vel = self._get_base_state()
        qpos, qvel = self._get_joint_state()

        return {
            "base_orientation": base_orientation,
            "qpos": qpos,
            "base_angular_vel": base_angular_vel,
            "qvel": qvel,
        }

    def step(self, dt: float, q_ref: np.ndarray, dq_ref: np.ndarray, kps: np.ndarray, kds: np.ndarray) -> None:
        self.set_motor_commands(
            motor_indices=range(len(self.REAL_JOINT_NAME_ORDER)),
            positions=q_ref,
            velocities=dq_ref,
            kps=kps,
            kds=kds,
        )
        self.send_cmd(self.low_cmd)

        time_to_wait = dt - (time.perf_counter() - self.step_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        self.step_time = time.perf_counter()

    def should_quit(self) -> bool:
        if self.input_device is not None:
            return self.input_device.is_pressed(Button.select)
        return False

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(0.02)
        print("Successfully connected to the robot.")

    def set_motor_commands(self, motor_indices, positions, velocities, kps, kds):
        for i, motor_idx in enumerate(motor_indices):
            self.low_cmd.motor_cmd[motor_idx].q = positions[i]
            self.low_cmd.motor_cmd[motor_idx].dq = velocities[i]
            self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def _get_base_state(self):
        # h1_2 imu is in the torso
        # imu data needs to be transformed to the pelvis frame
        torso_id = 12
        torso = self.low_state.motor_state[torso_id]
        imu_state = self.low_state.imu_state
        return transform_imu_data(
            waist_yaw=torso.q,
            waist_yaw_omega=torso.dq,
            imu_quat=imu_state.quaternion,
            imu_omega=imu_state.gyroscope,
        )

    def _get_joint_state(self):
        num_joints = len(self.REAL_JOINT_NAME_ORDER)
        qpos = np.empty(num_joints)
        qvel = np.empty(num_joints)
        for i in range(num_joints):
            qpos[i] = self.low_state.motor_state[i].q
            qvel[i] = self.low_state.motor_state[i].dq

        return (qpos, qvel)

    def init_cmd_hg(self, cmd: LowCmdHG, mode_machine: int, mode_pr: int):
        cmd.mode_machine = mode_machine
        cmd.mode_pr = mode_pr
        for i in range(len(cmd.motor_cmd)):
            cmd.motor_cmd[i].mode = 1
        self.enter_zero_torque_state()

    def enter_zero_torque_state(self):
        print("Entering zero torque state.")
        self.set_motor_commands(
            motor_indices=range(self.num_joints_total),
            positions=np.zeros(self.num_joints_total),
            velocities=np.zeros(self.num_joints_total),
            kps=np.zeros(self.num_joints_total),
            kds=np.zeros(self.num_joints_total),
        )
        self.send_cmd(self.low_cmd)

    def enter_damping_state(self):
        print("Entering damping state.")
        self.set_motor_commands(
            motor_indices=range(self.num_joints_total),
            positions=np.zeros(self.num_joints_total),
            velocities=np.zeros(self.num_joints_total),
            kps=np.zeros(self.num_joints_total),
            kds=8 * np.ones(self.num_joints_total),
        )
        self.send_cmd(self.low_cmd)

    def move_to_pos(self, joint_idx, pos, kps, kds, dt, duration):
        num_step = int(duration / dt)

        # Record the current pos
        init_dof_pos = np.array([self.low_state.motor_state[i].q for i in joint_idx])

        # Move to pos
        for i in range(num_step):
            alpha = i / num_step
            target_pos = init_dof_pos * (1 - alpha) + pos * alpha
            self.set_motor_commands(joint_idx, target_pos, np.zeros_like(target_pos), kps, kds)
            self.send_cmd(self.low_cmd)
            time.sleep(dt)
        self.set_motor_commands(joint_idx, pos, np.zeros_like(pos), kps, kds)
        self.send_cmd(self.low_cmd)

    def close(self):
        self.enter_damping_state()
