import numpy as np

import mujoco

from robot_deploy.robots.robot import Robot
from robot_deploy.simulator.sim_mujoco import MujocoSim


class ConfigError(Exception): ...


class H12Mujoco(MujocoSim, Robot):
    def __init__(self, scene_path, config):
        super().__init__(scene_path, config)
        self.set_config(config)

    def set_config(self, config: dict):
        self.decimation = config["mujoco"]["decimation"]

        joints = config["joints"]
        config_joint_names = [joint["name"] for joint in joints]

        num_joints = self.model.njnt - 1
        self.joint_kp = np.empty(num_joints)
        self.joint_kd = np.empty(num_joints)
        self.default_joint_pos = np.empty(num_joints)
        for joint_id in range(num_joints):
            # Joint is +1 because joint 0 is floating_base_joint
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id + 1)
            if joint_name not in config_joint_names:
                err_msg = f"Joint '{joint_name}' is not set up in the config file"
                raise ConfigError(err_msg)
            joint_config = joints[config_joint_names.index(joint_name)]
            self.joint_kp[joint_id] = joint_config["kp"]
            self.joint_kd[joint_id] = joint_config["kd"]
            self.default_joint_pos[joint_id] = joint_config["default_joint_pos"]

        enabled_joint_mujoco_idx = []
        for joint in joints:
            if not joint["enabled"]:
                continue
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint["name"])
            if joint_id == -1:
                err_msg = f"Joint '{joint['name']}' is enabled, but cannot be found in the model"
                raise ConfigError(err_msg)
            enabled_joint_mujoco_idx.append(joint_id - 1)  # -1 because 0 is floating_base_joint
        self.enabled_joint_mujoco_idx = np.array(enabled_joint_mujoco_idx)

    def get_robot_state(self):
        state = super().get_robot_state()
        state["qpos"] = state["qpos"][self.enabled_joint_mujoco_idx]
        state["qvel"] = state["qvel"][self.enabled_joint_mujoco_idx]
        return state

    def step(self, q_ref):
        q_whole = self.default_joint_pos.copy()
        q_whole[self.enabled_joint_mujoco_idx] = q_ref
        for _ in range(self.decimation):
            torques = self._pd_control(q_whole)
            self.sim_step(torques)

    def _pd_control(self, q_ref):
        state = super().get_robot_state()

        q_err = q_ref - state["qpos"]
        q_err_dot = np.zeros_like(q_ref) - state["qvel"]
        return self.joint_kp * q_err + self.joint_kd * q_err_dot
