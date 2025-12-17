import numpy as np

import mujoco

from robot_deploy.input_devices import Button, InputDevice
from robot_deploy.simulators import MujocoSim

from .robot import Robot


class H12Mujoco(MujocoSim, Robot):
    def __init__(self, config: dict, input_device: InputDevice | None = None) -> None:
        super().__init__(config["mujoco"], input_device)
        self.input_device = input_device
        self.sim_dt = config["mujoco"]["simulation_dt"]

    def get_joint_names(self) -> list[str]:
        return [
            # Joint is +1 because joint 0 is floating_base_joint
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id + 1)
            for joint_id in range(self.model.njnt - 1)
        ]

    def initialize(self) -> None:
        if self.input_device is not None:
            self.input_device.wait_for(Button.start)

    def step(self, dt: float, q_ref: np.ndarray, dq_ref: np.ndarray, kps: np.ndarray, kds: np.ndarray) -> None:
        nb_steps = int(dt / self.sim_dt)
        assert nb_steps > 0, "Simulation dt smaller than control dt"

        for _ in range(nb_steps):
            torques = self._pd_control(q_ref, dq_ref, kps, kds)
            self.sim_step(self.sim_dt, torques)

    def should_quit(self) -> bool:
        if self.input_device is not None:
            return self.input_device.is_pressed(Button.select)
        return self.current_time > self.episode_length

    def _pd_control(self, q_ref: np.ndarray, dq_ref: np.ndarray, kps: np.ndarray, kds: np.ndarray) -> np.ndarray:
        state = super().get_robot_state()

        q_err = q_ref - state["qpos"]
        q_err_dot = dq_ref - state["qvel"]
        return kps * q_err + kds * q_err_dot
