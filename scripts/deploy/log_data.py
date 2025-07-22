import argparse
import numpy as np
import struct
import time
import yaml
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    parser.add_argument("-o", "--output", type=str, default=None)
    return parser.parse_args()


class Logger:
    def __init__(self, output_name, config):
        ChannelFactoryInitialize(0, config["net_interface"])

        cmd_file = f"{output_name}_lowcmd.log" if output_name is not None else "lowcmd.log"
        state_file = f"{output_name}_lowstate.log" if output_name is not None else "lowstate.log"

        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.low_cmd_handler, 10)
        self.cmd_file = Path(cmd_file).open("ab")  # noqa: SIM115
        self.first_cmd = True

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)
        self.state_file = Path(state_file).open("ab")  # noqa: SIM115
        self.first_state = True

    def low_cmd_handler(self, msg: LowCmd_):
        if self.first_cmd:
            self.first_cmd = False
            self.cmd_file.write(struct.pack("i", len(msg.motor_cmd)))
        data = bytearray(8 * 4 * len(msg.motor_cmd))
        for i, cmd in enumerate(msg.motor_cmd):
            struct.pack_into("dddd", data, 8 * 4 * i, cmd.q, cmd.dq, cmd.kp, cmd.kd)
        self.cmd_file.write(data)

    def low_state_handler(self, msg: LowState_):
        if self.first_state:
            self.first_state = False
            self.state_file.write(struct.pack("i", len(msg.motor_state)))
        data = bytearray((7 + 2 * len(msg.motor_state)) * 8)
        struct.pack_into("dddd", data, 0, *msg.imu_state.quaternion)
        struct.pack_into("ddd", data, 4 * 8, *msg.imu_state.gyroscope)
        for i, state in enumerate(msg.motor_state):
            struct.pack_into("dd", data, (7 + 2 * i) * 8, state.q, state.dq)
        self.state_file.write(data)

    def close(self):
        self.cmd_file.close()
        self.state_file.close()


def load_cmd_log(path):
    with path.open("rb") as file:
        data = file.read()
    (nb_motors,) = struct.unpack_from("i", data)
    cmds = np.frombuffer(data[4:], dtype=np.float64)
    cmds = cmds.reshape((-1, 4 * nb_motors))
    return {
        "qpos": cmds[:, ::4],
        "qvel": cmds[:, 1::4],
        "kp": cmds[:, 2::4],
        "kd": cmds[:, 3::4],
    }


def load_state_log(path):
    with path.open("rb") as file:
        data = file.read()
    (nb_motors,) = struct.unpack_from("i", data)
    line_size = 7 + 2 * nb_motors
    states = np.frombuffer(data[4:], dtype=np.float64)
    states = states.reshape((-1, line_size))
    return {
        "base_orientation": states[:, :4],
        "base_angular_vel": states[:, 4:7],
        "qpos": states[:, 7::2],
        "qvel": states[:, 8::2],
    }


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger = Logger(args.output, config["real"])
    print("Logging commands and states")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stop logging")
    finally:
        logger.close()
