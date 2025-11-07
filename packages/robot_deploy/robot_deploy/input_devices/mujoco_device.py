import numpy as np

import glfw

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
        super().__init__()
        self.command = np.zeros(3)

    def get_command(self) -> np.ndarray:
        with self.lock:
            return self.command

    def wait_for(self, *buttons: Button) -> None:
        self.button_press = [False] * len(Button)
        super().wait_for(*buttons)

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
                self._press_button(BUTTON_KEYMAP[key])
