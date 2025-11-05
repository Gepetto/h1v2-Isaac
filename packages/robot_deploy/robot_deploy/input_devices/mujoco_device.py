import numpy as np
import threading

import mujoco
import mujoco.viewer

from .input_device import Button, InputDevice


class MujocoDevice(InputDevice):
    def __init__(self) -> None:
        self.command = np.zeros(3)
        self.key_pressed = set()
        self.lock = threading.Lock()

        if not hasattr(mujoco.viewer, "key_callbacks"):
            mujoco.viewer.key_callbacks = []  # type: ignore
        mujoco.viewer.key_callbacks.append(self.key_callback)  # type: ignore

    def get_command(self) -> np.ndarray:
        with self.lock:
            return self.command

    def is_pressed(self, *buttons: Button) -> bool:
        with self.lock:
            return any(button.lower() in self.key_pressed for button in buttons)

    def clear(self) -> None:
        with self.lock:
            self.key_pressed.clear()

    def key_callback(self, key) -> None:
        glfw = mujoco.glfw.glfw  # type: ignore
        with self.lock:
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
                case glfw.KEY_ENTER:
                    self.key_pressed.add("start")
                case glfw.KEY_ESCAPE:
                    self.key_pressed.add("select")
                case _:
                    self.key_pressed.add(glfw.get_key_name(key, 0))
