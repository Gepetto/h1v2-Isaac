import time
from pathlib import Path

from enum import Enum
import numpy as np
import yaml
from controllers.rl import RLPolicy
from robots.h12_real import H12Real, KeyMap


class DebugMode(Enum):
    FULL_MOVEMENT = 0
    PD = 1
    NO_MOVEMENT = 2


if __name__ == "__main__":
    debug = DebugMode.NO_MOVEMENT

    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Set up interface to real robot
    debug_robot = True if debug == DebugMode.NO_MOVEMENT else False
    robot = H12Real(config=config["real"], debug=debug_robot)

    # Load policy
    policy_path = str(Path(__file__).parent / "config" / "agent_model.onnx")
    policy_config_path = Path(__file__).parent / "config" / "env.yaml"
    with policy_config_path.open() as f:
        policy_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    policy = RLPolicy(policy_path, policy_config)

    if debug == DebugMode.FULL_MOVEMENT or debug == DebugMode.PD:
        robot.enter_zero_torque_state()
        robot.wait_for_button(KeyMap.start)

    if debug == DebugMode.FULL_MOVEMENT:
        robot.move_to_default_pos()
        robot.wait_for_button(KeyMap.A)

    while True:
        try:
            state = robot.get_robot_state()

            if debug == DebugMode.FULL_MOVEMENT:
                q_ref = policy.step(state)
                robot.step(q_ref)
            elif debug == DebugMode.PD:
                leg_joint2motor_idx = np.array([3])
                joint_pos = np.array([0.36])
                kp = np.array([10])
                kd = np.array([4])
                robot.set_motor_commands(leg_joint2motor_idx, joint_pos, kp, kd)
                robot.send_cmd(robot.low_cmd)
                time.sleep(0.02)
            elif debug == DebugMode.NO_MOVEMENT:
                print(state)
                time.sleep(0.5)
            if robot.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break # Emergency Stop

    if debug == DebugMode.FULL_MOVEMENT:
        robot.enter_damping_state()
    elif debug == DebugMode.PD:
        robot.enter_zero_torque_state()
    print("Exit")
