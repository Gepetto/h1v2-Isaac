import sys
from pathlib import Path

import numpy as np
import yaml
from biped_assets import SCENE_PATHS

sys.path.append("../")
from simulator.sim_mujoco import MujocoSim


class H12Mujoco(MujocoSim):
    def __init__(self, scene_path, config):
        super().__init__(scene_path, config)

        self.decimation = config["decimation"]

    def step(self, q_ref):
        for _ in range(self.decimation):
            torques = self._pd_control(q_ref)
            self.sim_step(torques)

    def _pd_control(self, q_ref):
        q_error = q_ref - self.data.qpos[7:]
        q_dot_error = np.zeros_like(q_ref) - self.data.qvel[6:]
        return self.kp * q_error + self.kd * q_dot_error


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]["12dof"]
    sim = H12Mujoco(scene_path, config["mujoco"])

    state = sim.get_robot_state()
    while True:
        sim.step(state["q_pos"])
