import time
import yaml
from pathlib import Path

from biped_assets import SCENE_PATHS
from controllers.rl import RLPolicy
from robots.h12_mujoco import H12Mujoco

if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Create unique log directory
    if config["mujoco"]["log_data"]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / "logs" / config["mujoco"]["experiment_name"] / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = None

    # Set up simulation
    scene_path = SCENE_PATHS["h12"]["27dof"]
    sim = H12Mujoco(scene_path, config)

    # Load policy
    policy_path = str(Path(__file__).parent / "config" / config["policy_name"])
    policy_config_path = Path(__file__).parent / "config" / config["policy_config_name"]
    with policy_config_path.open() as f:
        policy_config = yaml.load(f, Loader=yaml.UnsafeLoader)
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
