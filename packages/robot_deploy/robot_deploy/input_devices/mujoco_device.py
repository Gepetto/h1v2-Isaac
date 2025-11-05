import numpy as np
import threading

import glfw
import mujoco.viewer

from .input_device import Button, InputDevice

BUTTON_KEYMAP = {
    glfw.KEY_R: Button.R1,
    glfw.KEY_L: Button.L1,
    glfw.KEY_ENTER: Button.start,
    glfw.KEY_ESCAPE: Button.select,
    glfw.KEY_A: Button.A,
    glfw.KEY_B: Button.B,
    glfw.KEY_X: Button.X,
    glfw.KEY_Y: Button.Y,
}


class MujocoDevice(InputDevice):
    def __init__(self) -> None:
        self.command = np.zeros(3)
        self.key_pressed = set()
        self.lock = threading.Lock()

        if not hasattr(mujoco.viewer, "key_callbacks"):
            mujoco.viewer.key_callbacks = []  # type: ignore
        mujoco.viewer.key_callbacks.append(self.key_callback)  # type: ignore

    def get_command(self) -> np.ndarray:
        self.key_pressed.clear()
        with self.lock:
            return self.command

    def is_pressed(self, *buttons: Button) -> bool:
        with self.lock:
            return any(button in self.key_pressed for button in buttons)

    def key_callback(self, key) -> None:
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
            if key in BUTTON_KEYMAP:
                self.key_pressed.add(BUTTON_KEYMAP[key])
