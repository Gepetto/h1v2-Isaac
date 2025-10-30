import argparse
import time
import yaml
from pathlib import Path

from robot_assets import SCENE_PATHS
from robot_deploy.controllers.rl import RLPolicy
from robot_deploy.robots.h12_real import H12Real
from robot_deploy.robots.h12_mujoco import H12Mujoco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    parser.add_argument("--sim", action="store_true", help="Flag to run the policy in simulation")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Set up interface to real robot
    use_mujoco = config["real"]["use_mujoco"]
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

    robot = H12Mujoco(config=config) if args.sim else H12Real(config=config)
    policy = RLPolicy(policy_path, policy_config, log_data=config["mujoco"]["log_data"])

    robot.initialize()
    try:
        while True:
            state = robot.get_robot_state()

            command = robot.get_controller_command()
            q_ref = policy.step(state, command)
            robot.step(q_ref)

            if robot.should_quit():
                break

    except KeyboardInterrupt:
        print("Interruption")

    finally:
        robot.close(log_dir)
        policy.save_data(log_dir)
    print("Exit")
