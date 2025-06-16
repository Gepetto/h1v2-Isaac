from abc import ABC, abstractmethod
import numpy as np


class Robot(ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def apply_commands(self, u_ref, q_ref, v_ref) -> None:
        pass

    @abstractmethod
    def get_X_dX(self) -> (np.ndarray, np.ndarray):
        pass

    @abstractmethod
    def get_euler_orientation(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_IMU(self, q, v):
        pass
