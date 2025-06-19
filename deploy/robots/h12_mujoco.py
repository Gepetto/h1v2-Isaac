import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from biped_assets import SCENE_PATHS


class H1Mujoco:
    def __init__(
        self,
        scene_path,
        config,
    ):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = config["dt"]
        self.data = mujoco.MjData(self.model)
        self.real_time = config["real_time"]

        self.lock = threading.Lock()

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.kp = np.array(config["kp"])
        self.kd = np.array(config["kd"])
        self.decimation = config["decimation"]

        # Enable the weld constraint
        if config["fix_base"]:
            self.model.eq_active0[0] = 1
        else:
            self.model.eq_active0[0] = 0

        self.reset()

        if config["enable_GUI"]:
            thread = threading.Thread(target=self.run_render)
            thread.start()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def step(self, q_ref):
        for _ in range(self.decimation):
            step_start = time.perf_counter()
            torques = self._pd_control(q_ref)
            self._apply_torques(torques)
            self.lock.acquire()
            mujoco.mj_step(self.model, self.data)
            self.lock.release()

            if self.real_time:
                time_to_wait = max(0, step_start - time.perf_counter() + self.model.opt.timestep)
                time.sleep(time_to_wait)

    def _apply_torques(self, torques):
        self.data.ctrl[:] = torques

    def _pd_control(self, q_ref):
        q_error = q_ref - self.data.qpos[7:]
        q_dot_error = np.zeros_like(q_ref) - self.data.qvel[6:]
        return self.kp * q_error + self.kd * q_dot_error

    def get_robot_state(self):
        return {
            "base_orientation": self.data.qpos[3:7],
            "q_pos": self.data.qpos[7:],
            "base_angular_vel": self.data.qvel[3:6],
            "q_vel": self.data.qvel[6:],
        }

    def run_render(self):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        while viewer.is_running():
            self.lock.acquire()
            viewer.sync()
            self.lock.release()
            time.sleep(0.02)  # 50 Hz
        viewer.close()


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]
    sim = H1Mujoco(scene_path, config["mujoco"])

    state = sim.get_robot_state()
    while True:
        sim.step(state["q_pos"])
