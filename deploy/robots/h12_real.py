import argparse
import struct

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

        self.remote_controller = RemoteController()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        self.wait_for_low_state()
        self.init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def _set_config(self):
        self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def get_robot_state(self):
        q_pos, q_vel = self._get_joint_state()

        return {
            "base_orientation": self.data.qpos[3:7],
            "q_pos": q_pos,
            "base_angular_vel": self.data.qvel[3:6],
            "q_vel": q_vel,
        }

    def _get_joint_state(self):
        for i in range(len(self.config.leg_joint2motor_idx)):
            q_pos = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            q_vel = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        return (q_pos, q_vel)

    def enter_zero_torque_state():
        pass


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
