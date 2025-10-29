import numpy as np
from abc import ABC, abstractmethod


class Robot(ABC):
    @abstractmethod
    def set_config(self, config: dict) -> None:
        pass

    @abstractmethod
    def get_robot_state(self) -> dict:
        pass

    @abstractmethod
    def step(self, q_ref: np.ndarray) -> None:
        pass

