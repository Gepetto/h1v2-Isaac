import numpy as np
from abc import ABC, abstractmethod

from robot_deploy.input_devices import InputDevice


class Robot(ABC):
    @abstractmethod
    def __init__(self, config: dict, input_device: InputDevice | None = None) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def get_joint_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_robot_state(self) -> dict:
        pass

    @abstractmethod
    def step(self, dt: float, q_ref: np.ndarray, dq_ref: np.ndarray, kps: np.ndarray, kds: np.ndarray) -> None:
        pass

    @abstractmethod
    def should_quit(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
