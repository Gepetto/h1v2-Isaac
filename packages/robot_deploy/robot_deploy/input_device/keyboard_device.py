import numpy as np

import mujoco

from .input_device import Button, InputDevice


class KeyboardDevice(InputDevice):
    def __init__(self) -> None:
        self.command = np.zeros(3)
        self.key_pressed = set()

    def get_command(self) -> np.ndarray:
        return self.command

    def is_pressed(self, *buttons: Button) -> bool:
        is_pressed = any(button in self.key_pressed for button in buttons)
        self.key_pressed.clear()
        return is_pressed

    def key_callback(self, key) -> None:
        glfw = mujoco.glfw.glfw  # type: ignore
        match key:
            case glfw.KEY_UP | glfw.KEY_KP_8:
                self.command[0] += 0.1
            case glfw.KEY_DOWN | glfw.KEY_KP_5:
                self.command[0] -= 0.1
            case glfw.KEY_LEFT | glfw.KEY_KP_4:
                self.command[1] += 0.1
            case glfw.KEY_RIGHT | glfw.KEY_KP_6:
                self.command[1] -= 0.1
            case glfw.KEY_Z | glfw.KEY_KP_7:
                self.command[2] += 0.1
            case glfw.KEY_X | glfw.KEY_KP_9:
                self.command[2] -= 0.1
            case _:
                self.key_pressed.add(glfw.get_key_name(key, 0))
