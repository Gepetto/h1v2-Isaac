import numpy as np
import threading
import time

import mujoco
import mujoco.viewer

from robot_assets import SCENE_PATHS
from robot_deploy.utils.mj_logger import MJLogger


class ConfigError(Exception): ...


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
    def __init__(self, config: dict):
        mj_config = config["mujoco"]
        scene_path = SCENE_PATHS[mj_config["robot_name"]][mj_config["scene_name"]]

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = config["control_dt"] / mj_config["decimation"]
        self.current_time = 0
        self.episode_length = mj_config["episode_length"]
        self.data = mujoco.MjData(self.model)

        self.real_time = mj_config["real_time"]
        self.render_dt = mj_config["render_dt"]

        self.sim_lock = threading.Lock()

        self.log_data = mj_config["log_data"]
        if self.log_data:
            self.logger = MJLogger(self.model, self.data)
            self.logger.record_limits()

        # Enable the weld constraint
        self.model.eq_active0[0] = 1 if mj_config["fix_base"] else 0

        self.reset()

        if self.model.nu != self.model.njnt - 1:
            err_msg = "Not all joints are actuated"
            raise ConfigError(err_msg)

        self.ctrl_idx = [None] * self.model.nu
        for act_id in range(self.model.nu):
            if self.model.actuator_trntype[act_id] != mujoco.mjtTrn.mjTRN_JOINT:
                act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
                err_msg = f'Actuator "{act_name}" is not of transmission type JOINT'
                raise ConfigError(err_msg)

            jnt_id = self.model.actuator_trnid[act_id, 0]
            if self.ctrl_idx[jnt_id - 1] is not None:
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                act_name_1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.ctrl_idx[jnt_id - 1])
                act_name_2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
                err_msg = (
                    f'The "{act_name_1}" and "{act_name_2}" actuators are controlling the same joint ("{joint_name}")'
                )
                raise ConfigError(err_msg)
            self.ctrl_idx[jnt_id - 1] = act_id

        self.elastic_band_enabled = mj_config["elastic_band"]
        self.elastic_band = ElasticBand()
        self.band_attached_link = self.model.body("torso_link").id

        self.enable_GUI = mj_config["enable_GUI"]
        if self.enable_GUI:
            self.close_event = threading.Event()
            self.viewer_thread = threading.Thread(target=self.run_render, args=(self.close_event,))
            self.viewer_thread.start()

    def reset(self):
        with self.sim_lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def sim_step(self, torques):
        step_start = time.perf_counter()
        with self.sim_lock:
            self._apply_torques(torques)
            if self.log_data:
                self.logger.record_metrics(self.current_time)

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

    def close(self, log_dir=None):
        # Close Mujoco viewer if opened
        if self.enable_GUI:
            self.close_event.set()
            self.viewer_thread.join()

        if self.log_data and log_dir is not None:
            self.logger.save_data(log_dir)

    def _apply_torques(self, torques):
        self.data.ctrl[self.ctrl_idx] = torques

    def get_robot_state(self):
        with self.sim_lock:
            return {
                "base_orientation": self.data.qpos[3:7],
                "qpos": self.data.qpos[7:],
                "base_angular_vel": self.data.qvel[3:6],
                "qvel": self.data.qvel[6:],
            }

    def run_render(self, close_event):
        if not hasattr(mujoco.viewer, "key_callbacks"):
            mujoco.viewer.key_callbacks = []  # type: ignore

        def key_callback(key):
            for callback in mujoco.viewer.key_callbacks:  # type: ignore
                callback(key)

        mujoco.viewer.key_callbacks.append(self.elastic_band_callback)  # type: ignore
        with self.sim_lock:
            viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback)

        while viewer.is_running() and not close_event.is_set():
            with self.sim_lock:
                viewer.sync()

            time.sleep(self.render_dt)
        viewer.close()

    def elastic_band_callback(self, key):
        glfw = mujoco.glfw.glfw  # type: ignore
        match key:
            case glfw.KEY_B:
                self.elastic_band_enabled = not self.elastic_band_enabled
            case glfw.KEY_I:
                self.elastic_band.length += 0.1
            case glfw.KEY_K:
                self.elastic_band.length = max(0, self.elastic_band.length - 0.1)
