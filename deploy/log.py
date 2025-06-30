import struct
import time
from pathlib import Path

import numpy as np
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_


class Logger:
    def __init__(self, config):
        ChannelFactoryInitialize(0, config["net_interface"])

        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self.low_cmd_handler, 10)
        self.cmd_file = Path("lowcmd.log").open("ab")  # noqa: SIM115
        self.first_cmd = True

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)
        self.state_file = Path("lowstate.log").open("ab")  # noqa: SIM115
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
    array = np.frombuffer(data[4:], dtype=np.float64)
    return array.reshape((-1, 4 * nb_motors))


def load_state_log(path):
    with path.open("rb") as file:
        data = file.read()
    (nb_motors,) = struct.unpack_from("i", data)
    line_size = 7 + 2 * nb_motors
    array = np.frombuffer(data[4:], dtype=np.float64)
    return array.reshape((-1, line_size))


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger = Logger(config["real"])
    print("Logging commands and states")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stop logging")
    finally:
        logger.close()
