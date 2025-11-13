from __future__ import annotations

import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from robot_deploy.robots.robot import Robot

if TYPE_CHECKING:
    import numpy as np


class ConfigError(Exception): ...


class Policy(ABC):
    def __init__(self, _: Robot, policy_dir: Path, log_data: bool = False) -> None:
        self._load_config(policy_dir)
        self._get_policy_path(policy_dir)
        self.log_data = log_data

    @abstractmethod
    def step(self, state: dict, command: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def save_data(self, log_dir=None):
        pass

    def _load_config(self, policy_dir: Path):
        config_path = policy_dir / "env.yaml"
        if not config_path.exists():
            err_msg = f'No policy config `env.yaml` found in directory "{policy_dir}"'
            raise ConfigError(err_msg)

        with config_path.open() as f:
            self.config = yaml.load(f, Loader=yaml.UnsafeLoader)

    def _get_policy_path(self, policy_dir: Path) -> None:
        policy_paths = [path for path in policy_dir.iterdir() if path.is_file() and path.suffix in (".pt", ".onnx")]
        if len(policy_paths) != 1:
            if len(policy_paths) == 0:
                err_msg = f'No policy file found in directory "{policy_dir}" (no `.pt` or `.onnx` file)'
            else:
                err_msg = f'Multiple policy file found in directory "{policy_dir}": {", ".join(map(str, policy_paths))}'
            raise ConfigError(err_msg) from None
        self.policy_path = policy_paths[0]
