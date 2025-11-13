from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, unique
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@unique
class Button(Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    R1 = 4
    R2 = 5
    L1 = 6
    L2 = 7
    F1 = 8
    F2 = 9
    start = 10
    select = 11
    up = 12
    right = 13
    down = 14
    left = 15


class InputDevice(ABC):
    def __init__(self) -> None:
        self.bindings = [[] for _ in Button]
        self.lock = Lock()

    @abstractmethod
    def get_command(self) -> np.ndarray:
        pass

    @abstractmethod
    def is_pressed(self, *buttons: Button) -> bool:
        pass

    def bind(self, button: Button, callback: Callable[[], None]) -> None:
        self.bindings[button.value].append(callback)

    def wait_for(self, *buttons: Button) -> None:
        button_repr = " | ".join([f"'{button}'" for button in buttons])
        print(f"Waiting for button {button_repr}...")
        while not self.is_pressed(*buttons):
            time.sleep(0.02)

    def close(self) -> None:  # noqa
        pass
