import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from biped_assets import SCENE_PATHS

sys.path.append("../")
from simulator.sim_mujoco import MujocoSim


class H12Mujoco(MujocoSim):
    def __init__(self, scene_path, config):
        super().__init__(scene_path, config)

        self.decimation = config["mujoco"]["decimation"]

        joints = config["joints"]
        self.joint_kp = np.empty(len(joints))
        self.joint_kd = np.empty(len(joints))
        self.default_joint_pos = np.empty(len(joints))
        for joint in joints:
            sim_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"] + "_joint") - 1
            self.joint_kp[sim_joint_id] = joint["kp"]
            self.joint_kd[sim_joint_id] = joint["kd"]
            self.default_joint_pos[sim_joint_id] = joint["default_joint_pos"]

    def step(self, q_ref):
        q_whole = self.default_joint_pos.copy()
        q_whole[self.enabled_joint_mujoco_idx] = q_ref
        for _ in range(self.decimation):
            torques = self._pd_control(q_whole)
            self.sim_step(torques)

    def _pd_control(self, q_ref):
        with self.sim_lock:
            qpos = self.data.qpos[7:]
            qvel = self.data.qvel[6:]

        q_err = q_ref - qpos
        q_err_dot = np.zeros_like(q_ref) - qvel
        return self.joint_kp * q_err + self.joint_kd * q_err_dot


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    scene_path = SCENE_PATHS["h12"]["27dof"]
    sim = H12Mujoco(scene_path, config)

    state = sim.get_robot_state()
    while True:
        sim.step(state["qpos"])
