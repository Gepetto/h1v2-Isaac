import argparse
import time
from pathlib import Path

import yaml
from biped_assets import SCENE_PATHS
from controllers.rl import RLPolicy
from robots.h12_real import H12Real
from utils.remote_controller import KeyMap


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Set up interface to real robot
    use_mujoco = config["real"]["use_mujoco"]
    if use_mujoco:
        config["mujoco"]["scene_path"] = SCENE_PATHS["h12"]["27dof"]

    if use_mujoco and config["mujoco"]["log_data"]:
        # Create unique log directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / "logs" / config["mujoco"]["experiment_name"] / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = None

    # Load policy
    policy_dir: Path = Path(__file__).parent / "policies" / config["policy_name"]
    policy_path = str(next(filter(lambda file: file.name.endswith((".pt", ".onnx")), policy_dir.iterdir())))
    env_config_path = policy_dir / "env.yaml"

    with env_config_path.open() as f:
        policy_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    config.update(policy_config)

    robot = H12Real(config=config)
    policy = RLPolicy(policy_path, policy_config, log_data=config["mujoco"]["log_data"])

    if not use_mujoco:
        robot.enter_zero_torque_state()
        robot.wait_for_button(KeyMap.start)

        robot.move_to_default_pos()
        robot.wait_for_button(KeyMap.A)

    else:
        robot.set_init_state()

    try:
        while True:
            state = robot.get_robot_state()

            command = robot.get_controller_command()
            q_ref = policy.step(state, command)
            robot.step(q_ref)

            if robot.remote_controller.is_pressed(KeyMap.select):
                break

    except KeyboardInterrupt:
        print("Interruption")

    finally:
        robot.close(log_dir)
        policy.save_data(log_dir)
    print("Exit")
