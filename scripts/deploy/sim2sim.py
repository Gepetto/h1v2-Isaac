import argparse
import time
from pathlib import Path

import yaml
from biped_assets import SCENE_PATHS
from biped_deploy.controllers.rl import RLPolicy
from biped_deploy.robots.h12_mujoco import H12Mujoco


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", nargs="?", type=Path, default=None, help="Path to config file")
    return parser.parse_args()


if __name__ == "__main__":
    # Load config
    args = parse_args()
    config_path = args.config_path or Path(__file__).parent / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Create unique log directory
    if config["mujoco"]["log_data"]:
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

    # Set up simulation
    scene_path = SCENE_PATHS["h12"]["27dof"]
    sim = H12Mujoco(scene_path, config)
    policy = RLPolicy(policy_path, policy_config, log_data=config["mujoco"]["log_data"])

    try:
        while True:
            state = sim.get_robot_state()
            command = sim.get_controller_command()
            q_ref = policy.step(state, command)
            sim.step(q_ref)

            if sim.current_time > sim.episode_length:
                break

    except KeyboardInterrupt:
        pass

    finally:
        sim.close(log_dir)
        policy.save_data(log_dir)
