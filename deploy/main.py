from robots.h12_mujoco import H1Mujoco
from controllers.rl import RLPolicy

if __name__ == "__main__":
    scene_path = "/home/cperrot/h1v2-Isaac/urdfs/scene.xml"
    sim = H1Mujoco(scene_path, enable_GUI=False)

    policy_path = "/home/cperrot/h1v2-Isaac/deploy/config/agent_model.onnx"
    policy = RLPolicy(policy_path)

    while True:

        state = sim.get_robot_state()
        print(state)
        q_ref = policy.step(state)
        print(q_ref)
        sim.step(q_ref)

        break