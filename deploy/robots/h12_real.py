import struct
import time
from pathlib import Path

import numpy as np
import yaml
from biped_assets import SCENE_PATHS
from scipy.spatial.transform import Rotation as R
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


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


class H12Real:
    def __init__(self, config, scene_path=None, *, config_mujoco=None):
        ChannelFactoryInitialize(0, config["net_interface"])


        self.control_dt = config["control_dt"]

        self.leg_joint2motor_idx = np.array(config["leg_joint2motor_idx"])
        self.leg_kp = np.zeros_like(config["leg_kp"])
        self.leg_kd = np.zeros_like(config["leg_kd"])
        self.leg_default_joint_pos = np.array(config["leg_default_joint_pos"])

        self.arm_waist_joint2motor_idx = np.array(config["arm_waist_joint2motor_idx"])
        self.arm_waist_kp = np.zeros_like(config["arm_waist_kp"])
        self.arm_waist_kd = np.zeros_like(config["arm_waist_kd"])
        self.arm_waist_default_joint_pos = np.array(config["arm_waist_default_joint_pos"])

        self.remote_controller = RemoteController()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = 0  # MotorMode.PR in unitree code
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.use_mujoco = config_mujoco is not None
        if self.use_mujoco:
            assert scene_path is not None
            self.unitree = UnitreeSdk2Bridge(scene_path, config_mujoco)

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
        q_pos, q_vel = self._get_joint_state()

        return {
            "base_orientation": base_orientation,
            "q_pos": q_pos,
            "base_angular_vel": base_angular_vel,
            "q_vel": q_vel,
        }

    def _get_base_state(self):
        # TODO: use class variables to avoid memory allocation at runtime

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # h1_2 imu is in the torso
        # imu data needs to be transformed to the pelvis frame
        waist_yaw = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].q
        waist_yaw_omega = self.low_state.motor_state[self.arm_waist_joint2motor_idx[0]].dq
        quat, ang_vel = self.transform_imu_data(
            waist_yaw=waist_yaw,
            waist_yaw_omega=waist_yaw_omega,
            imu_quat=quat,
            imu_omega=ang_vel,
        )

        return (quat, ang_vel)

    def transform_imu_data(self, waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
        # TODO: extract this code to separater file and maybe replace with pinocchio for the conversion

        RzWaist = R.from_euler("z", waist_yaw).as_matrix()
        R_torso = R.from_quat(
            [imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]],
        ).as_matrix()
        R_pelvis = np.dot(R_torso, RzWaist.T)
        w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
        return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w

    def _get_joint_state(self):
        n = len(self.leg_joint2motor_idx)
        q_pos = np.empty(n)
        q_vel = np.empty(n)
        for i in range(n):
            q_pos[i] = self.low_state.motor_state[self.leg_joint2motor_idx[i]].q
            q_vel[i] = self.low_state.motor_state[self.leg_joint2motor_idx[i]].dq

        return (q_pos, q_vel)

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

        t_pose = self.get_t_pose()

        dof_idx = np.concatenate((self.leg_joint2motor_idx, self.arm_waist_joint2motor_idx), axis=0)
        kps = np.concatenate((self.leg_kp, self.arm_waist_kp), axis=0)
        kds = np.concatenate((self.leg_kd, self.arm_waist_kd), axis=0)
        default_pos = np.concatenate((self.leg_default_joint_pos, t_pose), axis=0)

        # Move legs
        self.move_to_pos(dof_idx, default_pos, kps, kds, 2)
        self.move_to_pos(
            self.arm_waist_joint2motor_idx,
            self.arm_waist_default_joint_pos,
            self.arm_waist_kp,
            self.arm_waist_kd,
            2,
        )
        print("Reached default pos state.")

    def get_t_pose(self):
        n = len(self.arm_waist_joint2motor_idx)
        curr_arm_waist_pos = np.empty(n)
        for i in range(n):
            curr_arm_waist_pos[i] = self.low_state.motor_state[self.arm_waist_joint2motor_idx[i]].q

        t_pose = curr_arm_waist_pos.copy()
        t_pose[list(self.arm_waist_joint2motor_idx).index(14)] = 0.6
        t_pose[list(self.arm_waist_joint2motor_idx).index(21)] = -0.6
        return t_pose

    def move_to_pos(self, joint_idx, pos, kp, kd, duration):
        num_step = int(duration / self.control_dt)

        # record the current pos
        init_dof_pos = np.zeros(len(joint_idx), dtype=np.float32)
        for i, dof_id in enumerate(joint_idx):
            init_dof_pos[i] = self.low_state.motor_state[dof_id].q

        # move to pos
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
            self.leg_joint2motor_idx,
            target_dof_pos,
            self.leg_kp,
            self.leg_kd,
        )
        self.set_motor_commands(
            self.arm_waist_joint2motor_idx,
            self.arm_waist_default_joint_pos,
            self.arm_waist_kp,
            self.arm_waist_kd,
        )

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.control_dt)

    def wait_for_button(self, button):
        button_name = next(k for k, v in KeyMap.__dict__.items() if v == button)
        print(f"Waiting to press '{button_name}'...")
        while self.remote_controller.button[button] != 1:
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def close(self):
        if self.use_mujoco:
            self.unitree.close()
        else:
            self.enter_damping_state()


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    if config["real"]["use_mujoco"]:
        scene_path = SCENE_PATHS["h12"]["27dof"]
        robot = H12Real(config["real"], config_mujoco=config["mujoco"], scene_path=scene_path)
    else:
        robot = H12Real(config["real"])

    robot.enter_zero_torque_state()
    robot.move_to_default_pos()

    state = robot.get_robot_state()
    while True:
        try:
            robot.step(state["q_pos"])
            # Press the select key to exit
            if robot.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    robot.enter_damping_state()
    robot.close()
    print("Exit")
