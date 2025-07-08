import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.append("../")
from utils.mj_logger import MJLogger


class ElasticBand:
    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0.0
        self.enable = True

    def advance(self, x, v):
        """
        Args:
            dx: desired position - current position
            v: current velocity
        """
        dx = self.point - x
        distance = np.linalg.norm(dx)
        direction = dx / distance
        v = np.dot(v, direction)
        return (self.stiffness * (distance - self.length) - self.damping * v) * direction

class MujocoSim:
    def __init__(self, scene_path, config):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = config["sim_dt"]
        self.current_time = 0
        self.episode_length = config["episode_length"]
        self.data = mujoco.MjData(self.model)

        self.real_time = config["real_time"]
        self.render_dt = config["render_dt"]

        self.sim_lock = threading.Lock()

        self.check_violations = config["check_violations"]
        if self.check_violations:
            self.logger = MJLogger(self.model, self.data)
            self.logger.record_limits()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Enable the weld constraint
        self.model.eq_active0[0] = 1 if config["fix_base"] else 0

        self.reset()

        self.enable_keyboard = config["enable_keyboard"]
        self.keyboard_lock = threading.Lock()
        self.controller_command = np.zeros(3)

        self.enable_GUI = config["enable_GUI"]
        if self.enable_GUI:
            self.close_event = threading.Event()
            self.viewer_thread = threading.Thread(target=self.run_render, args=(self.close_event,))
            self.viewer_thread.start()

        self.elastic_band_enabled = config["elastic_band"]
        if self.elastic_band_enabled:
            self.elastic_band = ElasticBand()
            self.band_attached_link = self.model.body("torso_link").id

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def sim_step(self, torques):
        step_start = time.perf_counter()
        self._apply_torques(torques)
        if self.check_violations:
            self.logger.record_metrics(self.current_time)

        with self.sim_lock:
            mujoco.mj_step(self.model, self.data)

        if self.elastic_band_enabled:
            self.data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.advance(
                self.data.qpos[:3],
                self.data.qvel[:3],
            )

        if self.real_time:
            time_to_wait = max(0, step_start - time.perf_counter() + self.model.opt.timestep)
            time.sleep(time_to_wait)

        self.current_time += self.model.opt.timestep

    def get_controller_command(self):
        with self.keyboard_lock:
            return self.controller_command

    def close(self, log_dir=None):
        # Close Mujoco viewer if opened
        if self.enable_GUI:
            self.close_event.set()
            self.viewer_thread.join()

        if self.check_violations and log_dir is not None:
            self.logger.save_data(log_dir)

    def _apply_torques(self, torques):
        self.data.ctrl[:] = torques

    def get_robot_state(self):
        with self.sim_lock:
            return {
                "base_orientation": self.data.qpos[3:7],
                "qpos": self.data.qpos[7:],
                "base_angular_vel": self.data.qvel[3:6],
                "qvel": self.data.qvel[6:],
            }

    def run_render(self, close_event):
        key_cb = self.key_callback if self.enable_keyboard else None
        viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb)
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
