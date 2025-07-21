import sys
import time
from pathlib import Path

import numpy as np
import yaml
from biped_assets import SCENE_PATHS
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

from .unitree_sdk2py_bridge import UnitreeSdk2Bridge

sys.path.append("../")
from utils.remote_controller import KeyMap, RemoteController
from utils.rotation import transform_imu_data


class ConfigError(Exception): ...


class H12Real:
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

    def __init__(self, config):
        ChannelFactoryInitialize(0, config["real"]["net_interface"])

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

        self.remote_controller = RemoteController()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = 0  # MotorMode.PR in unitree code
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.use_mujoco = config["real"]["use_mujoco"]
        if self.use_mujoco:
            self.unitree = UnitreeSdk2Bridge(config)

        # Wait for the subscriber to receive data
        self.wait_for_low_state()

        self.num_joints_total = len(self.low_cmd.motor_cmd)

        self.init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def set_motor_commands(self, motor_indices, positions, kps, kds):
        for i, motor_idx in enumerate(motor_indices):
            self.low_cmd.motor_cmd[motor_idx].q = positions[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def get_controller_command(self):
        if self.use_mujoco:
            command = self.unitree.get_controller_command()
        else:
            command = [self.remote_controller.ly, -self.remote_controller.lx, -self.remote_controller.rx]
        return np.clip(np.array(command), -1, 1)

    def get_robot_state(self):
        base_orientation, base_angular_vel = self._get_base_state()
        qpos, qvel = self._get_joint_state()

        return {
            "base_orientation": base_orientation,
            "qpos": qpos,
            "base_angular_vel": base_angular_vel,
            "qvel": qvel,
        }

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
        self.set_motor_commands(
            motor_indices=range(self.num_joints_total),
            positions=np.zeros(self.num_joints_total),
            kps=np.zeros(self.num_joints_total),
            kds=np.zeros(self.num_joints_total),
        )

    def set_init_state(self):
        if not self.use_mujoco:
            return

        self.set_motor_commands(
            motor_indices=range(len(self.REAL_JOINT_NAME_ORDER)),
            positions=self.default_joint_pos,
            kps=self.joint_kp,
            kds=self.joint_kd,
        )

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

    def move_to_default_pos(self):
        print("Moving to default pos.")

        leg_joint_idx = np.arange(0, 12)
        arm_joint_idx = np.arange(12, 27)

        # First, set legs and raise shoulders to avoid hitting itself
        # t_pose = np.array([motor.q for motor in self.low_state.motor_state])
        t_pose = np.array([self.low_state.motor_state[i].q for i in range(27)])
        t_pose[leg_joint_idx] = self.default_joint_pos[leg_joint_idx]
        t_pose[self.REAL_JOINT_NAME_ORDER.index("left_shoulder_roll")] = 0.6
        t_pose[self.REAL_JOINT_NAME_ORDER.index("right_shoulder_roll")] = -0.6
        self.move_to_pos(range(27), t_pose, 2)

        # Then set up the default position
        self.move_to_pos(arm_joint_idx, self.default_joint_pos[arm_joint_idx], 2)

        print("Reached default pos state.")

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

    def step(self, target_dof_pos):
        self.set_motor_commands(
            self.enabled_joint_idx,
            target_dof_pos,
            self.joint_kp[self.enabled_joint_idx],
            self.joint_kd[self.enabled_joint_idx],
        )
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)

    def wait_for_button(self, button):
        button_name = next(k for k, v in KeyMap.__dict__.items() if v == button)
        print(f"Waiting to press '{button_name}'...")
        while not self.remote_controller.is_pressed(button):
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def close(self, log_dir=None):
        if self.use_mujoco:
            self.unitree.close(log_dir)
        else:
            self.enter_damping_state()


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    if config["real"]["use_mujoco"]:
        config["mujoco"]["scene_path"] = SCENE_PATHS["h12"]["27dof"]
    robot = H12Real(config["real"])

    robot.enter_zero_torque_state()
    robot.move_to_default_pos()

    state = robot.get_robot_state()
    while True:
        try:
            robot.step(state["qpos"])
            # Press the select key to exit
            if robot.remote_controller.is_pressed(KeyMap.select):
                break
        except KeyboardInterrupt:
            break

    robot.enter_damping_state()
    robot.close()
    print("Exit")
