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

    def set_config(self, config: dict):
        self.control_dt = config["control_dt"]

        joints = config["joints"]
        config_joint_names = [joint["name"] for joint in joints]

        num_joints = len(self.REAL_JOINT_NAME_ORDER)
        self.joint_kp = np.empty(num_joints)
        self.joint_kd = np.empty(num_joints)
        self.default_joint_pos = np.empty(num_joints)
        for joint_id in range(num_joints):
            joint_name = self.REAL_JOINT_NAME_ORDER[joint_id]
            if joint_name not in config_joint_names:
                err_msg = f"Joint '{joint_name}' is not set up in the config file"
                raise ConfigError(err_msg)
            joint_config = joints[config_joint_names.index(joint_name)]
            self.joint_kp[joint_id] = joint_config["kp"]
            self.joint_kd[joint_id] = joint_config["kd"]
            self.default_joint_pos[joint_id] = joint_config["default_joint_pos"]

        self.enabled_joint_idx = []
        for joint in joints:
            if not joint["enabled"]:
                continue
            if joint["name"] not in self.REAL_JOINT_NAME_ORDER:
                err_msg = f"Joint '{joint['name']}' is enabled, but cannot be found in the model"
                raise ConfigError(err_msg)
            self.enabled_joint_idx.append(self.REAL_JOINT_NAME_ORDER.index(joint["name"]))

    def initialize(self) -> None:
        if self.input_device is not None:
            self.input_device.wait_for(Button.start)

        print("Moving to default pos.")

        leg_joint_idx = np.arange(0, 12)
        arm_joint_idx = np.arange(12, 27)

        # First, set legs and raise shoulders to avoid hitting itself
        # t_pose = np.array([motor.q for motor in self.low_state.motor_state])
        t_pose = np.array([self.low_state.motor_state[i].q for i in range(27)])
        t_pose[leg_joint_idx] = self.default_joint_pos[leg_joint_idx]
        t_pose[self.REAL_JOINT_NAME_ORDER.index("left_shoulder_roll_joint")] = 0.6
        t_pose[self.REAL_JOINT_NAME_ORDER.index("right_shoulder_roll_joint")] = -0.6
        self.move_to_pos(range(27), t_pose, 2)

        # Then set up the default position
        self.move_to_pos(arm_joint_idx, self.default_joint_pos[arm_joint_idx], 2)

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

    def step(self, q_ref):
        self.set_motor_commands(
            self.enabled_joint_idx,
            q_ref,
            self.joint_kp[self.enabled_joint_idx],
            self.joint_kd[self.enabled_joint_idx],
        )
        self.send_cmd(self.low_cmd)

        time_to_wait = self.control_dt - (time.perf_counter() - self.step_time)
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
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def set_motor_commands(self, motor_indices, positions, kps, kds):
        for i, motor_idx in enumerate(motor_indices):
            self.low_cmd.motor_cmd[motor_idx].q = positions[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
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
        n = len(self.enabled_joint_idx)
        qpos = np.empty(n)
        qvel = np.empty(n)
        for i in range(n):
            qpos[i] = self.low_state.motor_state[self.enabled_joint_idx[i]].q
            qvel[i] = self.low_state.motor_state[self.enabled_joint_idx[i]].dq

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
            kps=np.zeros(self.num_joints_total),
            kds=np.zeros(self.num_joints_total),
        )
        self.send_cmd(self.low_cmd)

    def enter_damping_state(self):
        print("Entering damping state.")
        self.set_motor_commands(
            motor_indices=range(self.num_joints_total),
            positions=np.zeros(self.num_joints_total),
            kps=np.zeros(self.num_joints_total),
            kds=8 * np.ones(self.num_joints_total),
        )
        self.send_cmd(self.low_cmd)

    def move_to_pos(self, joint_idx, pos, duration):
        num_step = int(duration / self.control_dt)
        kp = self.joint_kp[joint_idx]
        kd = self.joint_kd[joint_idx]

        # Record the current pos
        init_dof_pos = np.array([self.low_state.motor_state[i].q for i in joint_idx])

        # Move to pos
        for i in range(num_step):
            alpha = i / num_step
            target_pos = init_dof_pos * (1 - alpha) + pos * alpha
            self.set_motor_commands(joint_idx, target_pos, kp, kd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

        self.set_motor_commands(joint_idx, pos, kp, kd)
        self.send_cmd(self.low_cmd)

    def close(self):
        self.enter_damping_state()
