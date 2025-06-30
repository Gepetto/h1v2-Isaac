import time
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from biped_assets import SCENE_PATHS
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
    debug_robot = debug == DebugMode.NO_MOVEMENT
    if config["real"]["use_mujoco"]:
        scene_path = SCENE_PATHS["h12"]["27dof"]
        robot = H12Real(config=config["real"], use_mujoco=True, scene_path=scene_path, debug=debug_robot)
    else:
        robot = H12Real(config=config["real"], debug=debug_robot)

    # Load policy
    policy_path = str(Path(__file__).parent / "config" / "model.onnx")
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

    try:
        while True:
            state = robot.get_robot_state()

            if debug == DebugMode.FULL_MOVEMENT:
                command = robot.get_controller_command()
                q_ref = policy.step(state, command)
                robot.step(q_ref)

            elif debug == DebugMode.PD:
                joint_id = 3  # left knee
                default_joint_pos = config["real"]["leg_joint2motor_idx"][joint_id]
                leg_joint2motor_idx = np.array([joint_id])
                joint_pos = np.array([default_joint_pos])
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
        print("Interruption")
    except Exception as err:
        print("Error:", err)
    finally:
        if debug == DebugMode.FULL_MOVEMENT:
            robot.enter_damping_state()
        elif debug == DebugMode.PD:
            robot.enter_zero_torque_state()
        print("Exit")
