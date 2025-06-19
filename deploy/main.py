from pathlib import Path

import yaml
from biped_assets import SCENE_PATHS
from controllers.rl import RLPolicy
from robots.h12_mujoco import H1Mujoco

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]
    sim = H1Mujoco(scene_path, config["mujoco"])

    policy_path = str(Path(__file__).parent / "config" / "agent_model.onnx")
    policy = RLPolicy(policy_path, config["rl"])

    while True:
        state = sim.get_robot_state()
        q_ref = policy.step(state)
        sim.step(q_ref)
