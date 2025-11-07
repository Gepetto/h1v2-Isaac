import numpy as np
import time

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

BUTTON_RESET_TIME = 0.1


class MujocoDevice(InputDevice):
    def __init__(self) -> None:
        super().__init__()
        self.button_press = [False] * len(Button)
        self.command = np.zeros(3)
        self.last_press_times = [time.perf_counter()] * len(Button)

    def get_command(self) -> np.ndarray:
        with self.lock:
            return self.command

    def is_pressed(self, *buttons: Button) -> bool:
        with self.lock:
            # Reset all buttons that have been pressed more than BUTTON_RESET_TIME ago
            press_time = time.perf_counter()
            for i in range(len(Button)):
                if self.button_press[i] and press_time - self.last_press_times[i] > BUTTON_RESET_TIME:
                    self.button_press[i] = False
        return any(self.button_press[button.value] for button in buttons)

    def key_callback(self, key) -> None:
        callback_fns = []
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
                button = BUTTON_KEYMAP[key]
                self.button_press[button.value] = True
                self.last_press_times[button.value] = time.perf_counter()
                callback_fns = self.bindings[button.value]

        # Run callbacks after releasing the lock
        for callback in callback_fns:
            callback()
