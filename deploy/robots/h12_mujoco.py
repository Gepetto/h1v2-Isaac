import mujoco
import mujoco.viewer
import numpy as np

from h1_assets import SCENE_PATHS


class H1Mujoco:
    def __init__(
        self,
        scene_path,
        dt=1e-3,
        enable_GUI=False,
        fix_base=False,
    ):
        self.enable_GUI = enable_GUI

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.model.opt.integrator = 3
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        self.viewer = None

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.qpos_des = self.data.qpos.copy()
        self.qvel_des = self.data.qvel.copy()
        self.ctrl_ff = self.data.ctrl.copy()

        self._set_config()

        # Enable the weld constraint
        if fix_base:
            self.model.eq_active0[0] = 1
        else:
            self.model.eq_active0[0] = 0

        self.reset()

    def _set_config(self):
        self.kp = np.array([200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40])
        self.kd = np.array([2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2])

        self.decimation = 20

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def step(self, q_ref):
        for _ in range(self.decimation):
            torques = self._pd_control(q_ref)
            self._apply_torques(torques)
            mujoco.mj_step(self.model, self.data)
            if self.enable_GUI:
                self.render()

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
        # return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    scene_path = SCENE_PATHS["h12"]
    sim = H1Mujoco(scene_path, enable_GUI=True)

    state = sim.get_robot_state()
    while True:
        sim.step(state[7:19])
