from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

Button = str


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
