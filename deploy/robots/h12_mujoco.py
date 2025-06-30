import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from biped_assets import SCENE_PATHS

sys.path.append("../")
from simulator.sim_mujoco import MujocoSim


class H12Mujoco(MujocoSim):
    def __init__(self, scene_path, config):
        super().__init__(scene_path, config)

        self.decimation = config["decimation"]
        self.keyboard_lock = threading.Lock()
        self.enable_keyboard = config["enable_keyboard"]
        self.controller_command = np.zeros(3)

    def step(self, q_ref):
        for _ in range(self.decimation):
            torques = self._pd_control(q_ref)
            self.sim_step(torques)

    def get_controller_command(self):
        with self.keyboard_lock:
            return self.controller_command

    def _pd_control(self, q_ref):
        q_error = q_ref - self.data.qpos[7:]
        q_dot_error = np.zeros_like(q_ref) - self.data.qvel[6:]
        return self.kp * q_error + self.kd * q_dot_error

    def run_render(self, close_event):
        viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
        while viewer.is_running() and not close_event.is_set():
            with self.sim_lock:
                viewer.sync()
            time.sleep(self.render_dt)
        viewer.close()

    def key_callback(self, key):
        glfw = mujoco.glfw.glfw
        with self.keyboard_lock:
            if key == glfw.KEY_UP or key == glfw.KEY_KP_8:
                self.controller_command[0] += 0.1
            elif key == glfw.KEY_DOWN or key == glfw.KEY_KP_5:
                self.controller_command[0] -= 0.1
            elif key == glfw.KEY_LEFT or key == glfw.KEY_KP_4:
                self.controller_command[1] += 0.1
            elif key == glfw.KEY_RIGHT or key == glfw.KEY_KP_6:
                self.controller_command[1] -= 0.1
            elif key == glfw.KEY_Z or key == glfw.KEY_KP_7:
                self.controller_command[2] += 0.1
            elif key == glfw.KEY_X or key == glfw.KEY_KP_9:
                self.controller_command[2] -= 0.1
            elif key == glfw.KEY_B:
                self.elastic_band_enabled = not self.elastic_band_enabled
            elif key == glfw.KEY_I:
                self.elastic_band.length += 0.1
            elif key == glfw.KEY_K:
                self.elastic_band.length -= 0.1


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]["12dof"]
    sim = H12Mujoco(scene_path, config["mujoco"])

    state = sim.get_robot_state()
    while True:
        sim.step(state["q_pos"])
