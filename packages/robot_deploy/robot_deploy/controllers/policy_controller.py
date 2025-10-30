import numpy as np
import yaml
from pathlib import Path

from robot_deploy.controllers.rl import RLPolicy


class ConfigError(Exception): ...


class PolicyController:
    def __init__(self, config_path: Path, log_data: bool = False) -> None:
        if not config_path.exists():
            err_msg = f'Config file "{config_path}" does not exist'
            raise ConfigError(err_msg)
        with config_path.open() as file:
            self.config = yaml.safe_load(file)

        if isinstance(self.config["policy_names"], str):
            self.config["policy_names"] = [self.config["policy_names"]]

        if not self.config["policy_names"]:
            err_msg = "No policy given! Please specify at least one control policy in the config file"
            raise ConfigError(err_msg)

        self.policies = []
        self.policy_configs = []
        for policy_name in self.config["policy_names"]:
            policy_dir = config_path.parent / "policies" / policy_name
            try:
                policy_path = str(next(filter(lambda file: file.name.endswith((".pt", ".onnx")), policy_dir.iterdir())))
            except StopIteration:
                err_msg = f'No policy file found in directory "{policy_dir}" (no `.pt` or `.onnx` file)'
                raise ConfigError(err_msg) from None

            env_config_path = policy_dir / "env.yaml"
            if not env_config_path.exists():
                err_msg = f'No policy config `env.yaml` found in directory "{policy_dir}"'
                raise ConfigError(err_msg)

            with env_config_path.open() as f:
                policy_config = yaml.load(f, Loader=yaml.UnsafeLoader)

            self.policy_configs.append(policy_config)
            self.policies.append(RLPolicy(policy_path, policy_config, log_data=log_data))

        self.nb_policies = len(self.policies)
        self.policy_index = 0

    def get_config(self) -> dict:
        return self.config | self.policy_configs[self.policy_index]

    def select_next_policy(self) -> dict:
        self.policy_index = (self.policy_index + 1) % self.nb_policies
        return self.get_config()

    def select_prev_policy(self) -> dict:
        self.policy_index = (self.policy_index + self.nb_policies - 1) % self.nb_policies
        return self.get_config()

    def step(self, state: dict, command: np.ndarray) -> np.ndarray:
        return self.policies[self.policy_index].step(state, command)
