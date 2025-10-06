import numpy as np
import time
import yaml
from enum import Enum
from pathlib import Path

from robots.h12_real import H12Real
from utils.remote_controller import KeyMap


class DebugMode(Enum):
    NO_MOVEMENT = 0
    PD = 1


if __name__ == "__main__":
    mode = DebugMode.NO_MOVEMENT

    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Set up interface to real robot
    debug_robot = mode == DebugMode.NO_MOVEMENT
    robot = H12Real(config=config["real"])

    if mode == DebugMode.PD:
        robot.enter_zero_torque_state()
        robot.wait_for_button(KeyMap.start)

    try:
        while True:
            state = robot.get_robot_state()

            if mode == DebugMode.NO_MOVEMENT:
                print(state)
                time.sleep(0.5)

            elif mode == DebugMode.PD:
                joint_id = 3  # left knee
                default_joint_pos = config["real"]["leg_joint2motor_idx"][joint_id]
                leg_joint2motor_idx = np.array([joint_id])
                joint_pos = np.array([default_joint_pos])
                kp = np.array([10])
                kd = np.array([4])
                robot.set_motor_commands(leg_joint2motor_idx, joint_pos, kp, kd)
                robot.send_cmd(robot.low_cmd)
                time.sleep(0.02)

            if robot.remote_controller.is_pressed(KeyMap.select):
                break

    except KeyboardInterrupt:
        print("Interruption")
    except Exception as err:
        print("Error:", err)

    finally:
        if mode == DebugMode.PD:
            robot.enter_zero_torque_state()
    print("Exit")
