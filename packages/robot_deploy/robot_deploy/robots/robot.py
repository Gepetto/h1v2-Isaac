import numpy as np
from abc import ABC, abstractmethod


class Robot(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

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
