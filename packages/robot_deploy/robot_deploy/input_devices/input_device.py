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
    def __init__(self) -> None:
        self.button_press = [False] * len(Button)
        self.bindings = [[] for _ in Button]

        self.lock = Lock()

    @abstractmethod
    def get_command(self) -> np.ndarray:
        pass

    def is_pressed(self, *buttons: Button) -> bool:
        button_pressed = False
        with self.lock:
            for button in buttons:
                if self.button_press[button.value]:
                    button_pressed = True
                    self.button_press[button.value] = False
        return button_pressed

    def bind(self, button: Button, callback: Callable[[], None]) -> None:
        self.bindings[button.value].append(callback)

    def wait_for(self, *buttons: Button) -> None:
        button_repr = " | ".join([f"'{button}'" for button in buttons])
        print(f"Waiting for button {button_repr}...")
        while not self.is_pressed(*buttons):
            time.sleep(0.1)

    def _press_button(self, button) -> None:
        self.button_press[button.value] = True
        for callback in self.bindings[button.value]:
            callback()
