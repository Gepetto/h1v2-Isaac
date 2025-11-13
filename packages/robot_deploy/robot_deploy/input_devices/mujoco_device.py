import numpy as np
import time

import glfw

from .input_device import Button, InputDevice

BUTTON_KEYMAP = {
    glfw.KEY_J: Button.L1,
    glfw.KEY_K: Button.R1,
    glfw.KEY_I: Button.L2,
    glfw.KEY_O: Button.R2,
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
        self.command = np.zeros(3)
        self.last_press_times = [0.0] * len(Button)

    def get_command(self) -> np.ndarray:
        with self.lock:
            return self.command

    def is_pressed(self, *buttons: Button) -> bool:
        press_time = time.perf_counter()
        with self.lock:
            # Check if any button have been pressed less than BUTTON_RESET_TIME ago
            return any(press_time - self.last_press_times[button.value] < BUTTON_RESET_TIME for button in buttons)

    def get_button_repr(self, button: Button) -> str:
        button_inv_keymaps = {value: key for key, value in BUTTON_KEYMAP.items()}
        if button not in button_inv_keymaps:
            return button.name.capitalize()
        glfw_key = button_inv_keymaps[button]
        if glfw_key == glfw.KEY_ENTER:
            return "Enter"
        if glfw_key == glfw.KEY_ESCAPE:
            return "Escape"
        return chr(glfw_key)

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
                self.last_press_times[button.value] = time.perf_counter()
                callback_fns = self.bindings[button.value]

        # Run callbacks after releasing the lock
        for callback in callback_fns:
            callback()
