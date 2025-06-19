from pathlib import Path

from controllers.rl import RLPolicy
from h1_assets import SCENE_PATHS
from robots.h12_mujoco import H1Mujoco

if __name__ == "__main__":
    scene_path = SCENE_PATHS["h12"]
    sim = H1Mujoco(scene_path, enable_GUI=False)

    policy_path = Path(__file__).parent / "config" / "agent_model.onnx"
    policy = RLPolicy(policy_path)

    while True:

        state = sim.get_robot_state()
        q_ref = policy.step(state)
        sim.step(q_ref)