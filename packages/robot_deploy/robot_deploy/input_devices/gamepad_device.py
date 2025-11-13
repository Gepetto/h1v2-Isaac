import numpy as np
import time
from pathlib import Path
from threading import Event, Thread

import glfw

from .input_device import Button, InputDevice

GLFW_BUTTON_KEYMAP = {
    Button.A: glfw.GAMEPAD_BUTTON_A,
    Button.B: glfw.GAMEPAD_BUTTON_B,
    Button.X: glfw.GAMEPAD_BUTTON_X,
    Button.Y: glfw.GAMEPAD_BUTTON_Y,
    Button.L1: glfw.GAMEPAD_BUTTON_LEFT_BUMPER,
    Button.R1: glfw.GAMEPAD_BUTTON_RIGHT_BUMPER,
    Button.L2: glfw.GAMEPAD_AXIS_LEFT_TRIGGER,
    Button.R2: glfw.GAMEPAD_AXIS_RIGHT_TRIGGER,
    Button.select: glfw.GAMEPAD_BUTTON_BACK,
    Button.start: glfw.GAMEPAD_BUTTON_START,
    Button.up: glfw.GAMEPAD_BUTTON_DPAD_UP,
    Button.down: glfw.GAMEPAD_BUTTON_DPAD_DOWN,
    Button.left: glfw.GAMEPAD_BUTTON_DPAD_LEFT,
    Button.right: glfw.GAMEPAD_BUTTON_DPAD_RIGHT,
}

EVENT_LOOP_DT = 0.05


def is_button_pressed(state, button) -> bool:
    glfw_button = GLFW_BUTTON_KEYMAP.get(button)
    if glfw_button is None:
        return False

    # L2 and R2 are axes, we consider them pressed if they're not in neutral position
    if button in (Button.L2, Button.R2):
        return state.axes[glfw_button] > -1.0
    return state.buttons[glfw_button]


class GamepadDevice(InputDevice):
    def __init__(self) -> None:
        super().__init__()

        with (Path(__file__).parent / "gamecontrollerdb.txt").open() as file:
            gamepad_mappings = file.read()

        glfw.init()
        glfw.update_gamepad_mappings(gamepad_mappings)

        self.joystick_id = None
        for i in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(i) and glfw.joystick_is_gamepad(i):
                self.joystick_id = i
                break

        if self.joystick_id is None:
            err_msg = "No GLFW gamepad found"
            raise ConnectionError(err_msg)

        self.close_event = Event()
        thread = Thread(target=self._run_event_loop, args=(self.close_event,))
        thread.start()

    def get_command(self) -> np.ndarray:
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return np.zeros(3)

        lx = state.axes[glfw.GAMEPAD_AXIS_LEFT_X]
        ly = state.axes[glfw.GAMEPAD_AXIS_LEFT_Y]
        rx = state.axes[glfw.GAMEPAD_AXIS_RIGHT_X]

        # Invert LY to match Pygame convention (up is -1 in GLFW)
        return np.clip([-ly, -lx, -rx], -1, 1)

    def is_pressed(self, *buttons: Button) -> bool:
        state = glfw.get_gamepad_state(self.joystick_id)
        if not state:
            return False
        return any(is_button_pressed(state, button) for button in buttons)

    def close(self) -> None:
        self.close_event.set()

    def _run_event_loop(self, close_event: Event) -> None:
        last_states = dict.fromkeys(GLFW_BUTTON_KEYMAP, False)
        while not close_event.is_set():
            state = glfw.get_gamepad_state(self.joystick_id)
            if not state:
                return

            for button in GLFW_BUTTON_KEYMAP:
                # Check for a transition from released to pressed
                current_state = is_button_pressed(state, button)
                if current_state and not last_states[button]:
                    for callback in self.bindings[button.value]:
                        try:
                            callback()
                        except Exception:
                            return
                last_states[button] = current_state

            time.sleep(EVENT_LOOP_DT)
