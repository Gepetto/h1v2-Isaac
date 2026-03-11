import numpy as np
import time
import yaml
from datetime import datetime
from pathlib import Path

from robot_deploy.robots import H12Real


class JointIdentifier:
    def __init__(self, joint_names: list[str]) -> None:
        self.joint_names = joint_names
        self.nb_joints = len(joint_names)

        self.min = [10000 for _ in joint_names]
        self.max = [-10000 for _ in joint_names]
        for joint_name, min_value, max_value in zip(self.joint_names, self.min, self.max, strict=True):
            line = f"{joint_name:>30} ∈ [{min_value:>6.3f}, {max_value:>6.3f}]"
            print(line)

    def observe(self, qpos: np.ndarray) -> None:
        for i, value in enumerate(qpos):
            self.min[i] = min(self.min[i], value)
            self.max[i] = max(self.max[i], value)

    def display(self) -> None:
        # \033[<N>A : Moves the cursor up N lines
        print(f"\033[{self.nb_joints}A", end="")

        for joint_name, min_value, max_value in zip(self.joint_names, self.min, self.max, strict=True):
            # \033[K : ANSI Code to Erase Line from Cursor to End of Line
            line = f"{joint_name:>30} ∈ [{min_value:>6.3f}, {max_value:>6.3f}]\033[K"
            print(line)

        print(flush=True, end="")

    def save(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "ranges.txt"
        with path.open("w") as file:
            for joint_name, min_value, max_value in zip(self.joint_names, self.min, self.max, strict=True):
                print(f"{joint_name:>30} ∈ [{min_value:>6.3f}, {max_value:>6.3f}]", file=file)


def main() -> None:
    config_path = Path(__file__).parent / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)
    config["mujoco"]["print_contacts"] = False

    robot = H12Real(config)

    joint_names = robot.get_joint_names()
    joint_identifier = JointIdentifier(joint_names)

    print("Start joint identification")
    try:
        while True:
            qpos = robot.get_robot_state()["qpos"]
            joint_identifier.observe(qpos)
            joint_identifier.display()
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        robot.close()
        log_path = Path("logs", "joint_identification", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print("Saving joint identification results to", log_path)
        joint_identifier.save(log_path)

    print("Exit")


if __name__ == "__main__":
    main()
