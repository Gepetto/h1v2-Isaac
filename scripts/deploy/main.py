import argparse
import yaml
from pathlib import Path

from robot_deploy.controllers.policy_controller import PolicyController
from robot_deploy.input_devices import Button, GamepadDevice, MujocoDevice, UnitreeRemoteDevice
from robot_deploy.robots.h12_mujoco import H12Mujoco
from robot_deploy.robots.h12_real import H12Real
from robot_deploy.simulators.dds_mujoco import DDSToMujoco


class ConfigError(Exception): ...


def load_config(path: Path) -> dict:
    if not path.exists():
        err_msg = f'Config file "{path}" does not exist'
        raise ConfigError(err_msg)
    with path.open() as file:
        config = yaml.safe_load(file)

    if not config["policy_names"]:
        err_msg = "No policy given! Please specify at least one control policy in the config file"
        raise ConfigError(err_msg)

    if isinstance(config["policy_names"], str):
        config["policy_names"] = [config["policy_names"]]

    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    parser.add_argument("--policy_dir", nargs="?", type=Path, default=None, help="Path to policy directory")
    parser.add_argument("--sim", action="store_true", help="Flag to run the policy in simulation")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"
    policy_dir = args.policy_dir or config_path.parent / "policies"
    config = load_config(config_path)

    use_bridge = not args.sim and config["real"]["use_mujoco"]
    if not (args.sim or use_bridge):
        input_device = UnitreeRemoteDevice(config["real"]["net_interface"])
    else:
        try:
            input_device = GamepadDevice()
        except ConnectionError:
            input_device = MujocoDevice()
    if use_bridge:
        simulator = DDSToMujoco(config, 0.001, input_device)

    robot = H12Mujoco(config, input_device) if args.sim else H12Real(config, input_device)
    policy_controller = PolicyController(robot, policy_dir, config["policy_names"])

    input_device.bind(Button.L1, policy_controller.select_prev_policy)
    input_device.bind(Button.R1, policy_controller.select_next_policy)

    try:
        robot.initialize()

        print("Start control")
        while True:
            command = input_device.get_command()
            state = robot.get_robot_state()

            dt, q_ref, kps, kds = policy_controller.step(state, command)
            robot.step(dt, q_ref, kps, kds)

            if robot.should_quit():
                break

    except KeyboardInterrupt:
        print("Interruption")

    finally:
        input_device.close()
        robot.close()
        if use_bridge:
            simulator.close()
    print("Exit")
