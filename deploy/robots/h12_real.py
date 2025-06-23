import argparse
import struct
import time

import numpy as np
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
    def __init__(self, net_interface):
        ChannelFactoryInitialize(0, net_interface)

        self._set_config()

        self.remote_controller = RemoteController()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = 0  # MotorMode.PR in unitree code
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        # Wait for the subscriber to receive data
        self.wait_for_low_state()

        self.num_joints_total = len(self.low_cmd.motor_cmd)

        # Initialize the command msg
        self.init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def _set_config(self):
        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.kps = [200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40]
        self.kds = [2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2]
        self.default_angles = np.array(
            [
                0,
                -0.16,
                0.0,
                0.36,
                -0.2,
                0.0,
                0,
                -0.16,
                0.0,
                0.36,
                -0.2,
                0.0,
            ]
        )

        self.arm_waist_joint2motor_idx = [
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
        ]
        self.arm_waist_kps = [
            300,
            120,
            120,
            120,
            80,
            80,
            80,
            80,
            120,
            120,
            120,
            80,
            80,
            80,
            80,
        ]

        self.arm_waist_kds = [3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1]

        self.arm_waist_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.control_dt = 0.02

    def LowStateHgHandler(self, msg: LowStateHG):
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
        return np.array(
            [
                self.remote_controller.ly,
                self.remote_controller.lx * -1,
                self.remote_controller.rx * -1,
            ]
        )

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
        waist_yaw_omega = self.low_state.motor_state[
            self.arm_waist_joint2motor_idx[0]
        ].dq
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
        for i in range(len(self.leg_joint2motor_idx)):
            q_pos = self.low_state.motor_state[self.leg_joint2motor_idx[i]].q
            q_vel = self.low_state.motor_state[self.leg_joint2motor_idx[i]].dq

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
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.set_motor_commands(
                motor_indices=range(self.num_joints_total),
                positions=np.zeros(self.num_joints_total),
                kps=np.zeros(self.num_joints_total),
                kds=np.zeros(self.num_joints_total),
            )
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

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
        total_time = 2  # move time 2s
        num_step = int(total_time / self.control_dt)

        dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        kps = self.kps + self.arm_waist_kps
        kds = self.kds + self.arm_waist_kds
        default_pos = np.concatenate(
            (self.default_angles, self.arm_waist_target), axis=0
        )

        # record the current pos
        init_dof_pos = np.zeros(self.num_joints_total, dtype=np.float32)
        for i, dof_id in enumerate(dof_idx):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            target_pos = init_dof_pos * (1 - alpha) + default_pos * alpha
            self.set_motor_commands(dof_idx, target_pos, kps, kds)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self.set_motor_commands(
                self.leg_joint2motor_idx,
                self.default_angles,
                self.kps,
                self.kds,
            )
            self.set_motor_commands(
                self.arm_waist_joint2motor_idx,
                self.arm_waist_target,
                self.arm_waist_kps,
                self.arm_waist_kds,
            )
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def step(self, target_dof_pos):
        self.set_motor_commands(
            self.leg_joint2motor_idx,
            target_dof_pos,
            self.kps,
            self.kds,
        )
        self.set_motor_commands(
            self.arm_waist_joint2motor_idx,
            self.arm_waist_target,
            self.arm_waist_kps,
            self.arm_waist_kds,
        )

        # send the command
        self.send_cmd(self.low_cmd)

        time.sleep(self.control_dt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()

    robot = H12Real(args.net)

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
    print("Exit")
