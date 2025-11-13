import numpy as np
from pathlib import Path

from robot_deploy.controllers.policy import Policy
from robot_deploy.controllers.rl_policy import RLPolicy
from robot_deploy.robots.robot import Robot


class ConfigError(Exception): ...


class PolicyController:
    def __init__(self, robot: Robot, policy_dir: Path, policy_names: list[str], log_data: bool = False) -> None:
        self.policy_names = policy_names
        self.policies: list[Policy] = []
        for policy_name in policy_names:
            policy = RLPolicy(robot, policy_dir / policy_name, log_data=log_data)
            self.policies.append(policy)

        self.nb_policies = len(self.policies)
        self.curr_policy_index = 0
        self.next_policy_index = 0

        self.total_merging_steps = 200
        self.curr_merging_step = 0

    def select_next_policy(self):
        self.next_policy_index = (self.next_policy_index + 1) % self.nb_policies
        if self.next_policy_index != self.curr_policy_index:
            self.curr_merging_step = self.total_merging_steps
        print(f"Transitioning to policy {self.policy_names[self.next_policy_index]}...")

    def select_prev_policy(self):
        self.next_policy_index = (self.next_policy_index + self.nb_policies - 1) % self.nb_policies
        if self.next_policy_index != self.curr_policy_index:
            self.curr_merging_step = self.total_merging_steps
        print(f"Transitioning to policy {self.policy_names[self.next_policy_index]}...")

    def step(self, state: dict, command: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.curr_merging_step == 0:
            return self.policies[self.curr_policy_index].step(state, command)

        dt1, q_ref1, dq_ref1, kps1, kds1 = self.policies[self.curr_policy_index].step(state, command)
        dt2, q_ref2, dq_ref2, kps2, kds2 = self.policies[self.next_policy_index].step(state, command)

        alpha = self.curr_merging_step / self.total_merging_steps

        q_ref = alpha * q_ref1 + (1 - alpha) * q_ref2
        dq_ref = alpha * dq_ref1 + (1 - alpha) * dq_ref2
        kps = alpha * kps1 + (1 - alpha) * kps2
        kds = alpha * kds1 + (1 - alpha) * kds2

        self.curr_merging_step -= 1
        if self.curr_merging_step == 0:
            self.curr_policy_index = self.next_policy_index
            print(f"Now running policy {self.policy_names[self.curr_policy_index]}!")

        return min(dt1, dt2), q_ref, dq_ref, kps, kds
