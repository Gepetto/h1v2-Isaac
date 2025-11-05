import argparse
from pathlib import Path

from robot_deploy.controllers.policy_controller import PolicyController
from robot_deploy.input_devices import MujocoDevice, UnitreeRemoteDevice
from robot_deploy.robots.h12_mujoco import H12Mujoco
from robot_deploy.robots.h12_real import H12Real


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    parser.add_argument("--sim", action="store_true", help="Flag to run the policy in simulation")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"

    # Set up interface to real robot
    policy_controller = PolicyController(config_path)
    config = policy_controller.get_config()

    if args.sim or config["real"]["use_mujoco"]:
        input_device = MujocoDevice()
    else:
        input_device = UnitreeRemoteDevice(config["real"]["net_interface"])
    robot = H12Mujoco(config, input_device) if args.sim else H12Real(config, input_device)

    robot.initialize()
    try:
        while True:
            state = robot.get_robot_state()

            command = input_device.get_command()
            q_ref = policy_controller.step(state, command)
            robot.step(q_ref)

            if robot.should_quit():
                break

    except KeyboardInterrupt:
        print("Interruption")

    finally:
        robot.close()
    print("Exit")
