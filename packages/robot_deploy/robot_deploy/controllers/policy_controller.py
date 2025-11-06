import numpy as np
from pathlib import Path

from robot_deploy.controllers.rl_policy import RLPolicy
from robot_deploy.robots.robot import Robot


class ConfigError(Exception): ...


class PolicyController:
    def __init__(self, robot: Robot, policy_dir: Path, policy_names: list[str], log_data: bool = False) -> None:
        self.policies = []
        for policy_name in policy_names:
            policy = RLPolicy(robot, policy_dir / policy_name, log_data=log_data)
            self.policies.append(policy)

        self.nb_policies = len(self.policies)
        self.policy_index = 0

    def select_next_policy(self):
        self.policy_index = (self.policy_index + 1) % self.nb_policies

    def select_prev_policy(self):
        self.policy_index = (self.policy_index + self.nb_policies - 1) % self.nb_policies

    def step(self, state: dict, command: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        return self.policies[self.policy_index].step(state, command)
