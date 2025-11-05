import argparse
from pathlib import Path

from robot_deploy.controllers.policy_controller import PolicyController
from robot_deploy.input_devices import Button, MujocoDevice, UnitreeRemoteDevice
from robot_deploy.robots.h12_mujoco import H12Mujoco
from robot_deploy.robots.h12_real import H12Real
from robot_deploy.simulators.dds_mujoco import DDSToMujoco


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
    use_bridge = not args.sim and config["real"]["use_mujoco"]

    if use_bridge:
        simulator = DDSToMujoco(config)

    robot = H12Mujoco(config) if args.sim else H12Real(config)
    input_device = MujocoDevice() if (args.sim or use_bridge) else UnitreeRemoteDevice(config["real"]["net_interface"])

    try:
        input_device.wait_for(Button.start)
        robot.initialize()
        input_device.wait_for(Button.A)

        print("Start control")
        while True:
            command = input_device.get_command()
            state = robot.get_robot_state()

            q_ref = policy_controller.step(state, command)
            robot.step(q_ref)

            if robot.should_quit() or input_device.is_pressed(Button.select):
                break

    except KeyboardInterrupt:
        print("Interruption")

    finally:
        robot.close()
        if use_bridge:
            simulator.close()
    print("Exit")
