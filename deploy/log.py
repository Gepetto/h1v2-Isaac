import logging
import time
from pathlib import Path

import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG


class Logger:
    def __init__(self, config):
        logging.basicConfig(filename="robot_log.txt", level=logging.DEBUG, format="%(asctime)s - %(message)s")

        ChannelFactoryInitialize(0, config["net_interface"])

        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmdHG)
        self.lowcmd_subscriber.Init(self.low_cmd_handler, 10)

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_cmd_handler(self, msg: LowCmdHG):
        msg_str = ";".join([f"{i}:({cmd.q},{cmd.dq},{cmd.kp},{cmd.kd})" for i, cmd in enumerate(msg.motor_cmd)])
        logging.debug("low_cmd:", msg_str)

    def low_state_handler(self, msg: LowStateHG):
        state = list(msg.imu_state.quaternion)
        state += list(msg.imu_state.gyroscope)
        state += [(state.q, state.dq) for state in msg.motor_state]
        msg_str = ";".join(map(str, state))
        logging.debug("low_state:", msg_str)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger = Logger(config["real"])
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stop logging")
