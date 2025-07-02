from pathlib import Path

import yaml
from biped_assets import SCENE_PATHS
from controllers.rl import RLPolicy
from robots.h12_real import H12Real
from utils.remote_controller import KeyMap

if __name__ == "__main__":
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    # Set up interface to real robot
    use_mujoco = config["real"]["use_mujoco"]
    if use_mujoco:
        scene_path = SCENE_PATHS["h12"]["27dof"]
        robot = H12Real(config=config["real"], config_mujoco=config["mujoco"], scene_path=scene_path)
    else:
        robot = H12Real(config=config["real"])

    # Load policy
    policy_path = str(Path(__file__).parent / "config" / "model.onnx")
    policy_config_path = Path(__file__).parent / "config" / "env.yaml"
    with policy_config_path.open() as f:
        policy_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    policy = RLPolicy(policy_path, policy_config)

    if not use_mujoco:
        robot.enter_zero_torque_state()
        robot.wait_for_button(KeyMap.start)

        robot.move_to_default_pos()
        robot.wait_for_button(KeyMap.A)

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
        robot.close()
    print("Exit")
