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


class H12Real:
    def __init__(self, config, scene_path=None, *, config_mujoco=None):
        ChannelFactoryInitialize(0, config["net_interface"])

        self.control_dt = config["control_dt"]

        self.leg_joint2motor_idx = np.array(config["leg_joint2motor_idx"])
        self.leg_kp = np.array(config["leg_kp"])
        self.leg_kd = np.array(config["leg_kd"])
        self.leg_default_joint_pos = np.array(config["leg_default_joint_pos"])

        self.arm_waist_joint2motor_idx = np.array(config["arm_waist_joint2motor_idx"])
        self.arm_waist_kp = np.array(config["arm_waist_kp"])
        self.arm_waist_kd = np.array(config["arm_waist_kd"])
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
        qpos, qvel = self._get_joint_state()

        return {
            "base_orientation": base_orientation,
            "qpos": qpos,
            "base_angular_vel": base_angular_vel,
            "qvel": qvel,
        }

    def _get_base_state(self):
        # h1_2 IMU is in the torso
        # IMU data needs to be transformed to the pelvis frame
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
        n = len(self.leg_joint2motor_idx)
        qpos = np.empty(n)
        qvel = np.empty(n)
        for i in range(n):
            qpos[i] = self.low_state.motor_state[self.leg_joint2motor_idx[i]].q
            qvel[i] = self.low_state.motor_state[self.leg_joint2motor_idx[i]].dq

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
        scene_path = SCENE_PATHS["h12"]["27dof"]
        robot = H12Real(config["real"], config_mujoco=config["mujoco"], scene_path=scene_path)
    else:
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
