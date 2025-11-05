import numpy as np
from abc import ABC, abstractmethod

from robot_deploy.input_devices import InputDevice


class Robot(ABC):
    def __init__(self, config: dict, input_device: InputDevice | None = None) -> None:
        self.set_config(config)
        self.input_device = input_device

    @abstractmethod
    def set_config(self, config: dict) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def get_robot_state(self) -> dict:
        pass

    @abstractmethod
    def step(self, q_ref: np.ndarray) -> None:
        pass

    @abstractmethod
    def should_quit(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
