import logging
import time
from pathlib import Path

import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG


class Logger:
    def __init__(self, config):
        logging.basicConfig(filename="robot_log.txt", level=logging.DEBUG, format="%(asctime)s - %(message)s")

        ChannelFactoryInitialize(0, config["net_interface"])

        self.low_cmd = unitree_hg_msg_dds__LowCmd_
        self.low_state = unitree_hg_msg_dds__LowState_

        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmdHG)
        self.lowcmd_subscriber.Init(self.low_cmd_handler, 10)

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_cmd_handler(self, msg: LowCmdHG):
        self.low_cmd = msg
        logging.debug(f"low_cmd: {self.low_cmd}")

    def low_state_handler(self, msg: LowStateHG):
        self.low_state = msg
        logging.debug(f"low_state: {self.low_state}")


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    logger = Logger(config["real"])
    while True:
        time.sleep(1)
        pass  # Keep the script running to continue logging
