from __future__ import annotations

from enum import Enum
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Button(Enum):
    A = 1
    B = 2
    X = 3
    Y = 4
    R1 = 5
    R2 = 6
    L1 = 7
    L2 = 8
    F1 = 9
    F2 = 10
    start = 11
    select = 12
    up = 13
    right = 14
    down = 15
    left = 16


class InputDevice(ABC):
    @abstractmethod
    def get_command(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_pressed(self, *buttons: Button) -> bool:
        pass

    def wait_for(self, button: Button) -> None:
        print(f"Waiting for button '{button}'...")
        while not self.is_pressed(button):
            time.sleep(0.1)
