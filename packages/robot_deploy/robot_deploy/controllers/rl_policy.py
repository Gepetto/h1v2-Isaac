import numpy as np
import torch
import yaml
from collections import deque
from pathlib import Path

import onnxruntime as ort

from robot_deploy.robots import Robot
from robot_deploy.utils.rl_logger import RLLogger

from .policy import Policy


class ConfigError(Exception): ...


class InferenceHandlerONNX:
    def __init__(self, policy_path):
        self.ort_sess = ort.InferenceSession(policy_path)
        self.input_name = self.ort_sess.get_inputs()[0].name

    def __call__(self, observations):
        observations_unsqueezed = np.expand_dims(observations, axis=0)
        actions = self.ort_sess.run(None, {self.input_name: observations_unsqueezed})[0]

        return actions.flatten()  # type: ignore


class InferenceHandlerTorch:
    def __init__(self, policy_path):
        self.policy = torch.jit.load(policy_path).to("cpu")

    def __call__(self, observations):
        obs_tensor = torch.from_numpy(observations).unsqueeze(0)
        actions = self.policy(obs_tensor).detach().numpy().squeeze()
        return actions


class ObservationHandler:
    def __init__(
        self,
        observations_func,
        observations_scale,
        history_length,
        default_joint_pos,
        commands_ranges,
        default_joint_vel=None,
    ):
        self.observations_func = [getattr(self, func_name) for func_name in observations_func]
        self.observations_scale = observations_scale
        self.history_length = history_length
        self.default_joint_pos = default_joint_pos
        self.commands_ranges = commands_ranges
        self.default_joint_vel = (
            default_joint_vel if default_joint_vel is not None else np.zeros_like(default_joint_pos)
        )
        self.observation_histories = {}
        self.command = np.array([0.0, 0.0, 0.0])
        self.counter = 0

        # Hard-coded parameters for the phase
        self.period = 0.8
        self.control_dt = 0.02

    def get_observations(self, state, actions, command):
        self.counter += 1

        self.state = state.copy()
        self.actions = actions.copy()
        if command is not None:
            self.command = command

        for i, element in enumerate(self.observations_func):
            if i not in self.observation_histories:
                self.observation_histories[i] = deque(maxlen=self.history_length)
                self.observation_histories[i].extend([element() * self.observations_scale[i]] * self.history_length)
            else:
                self.observation_histories[i].append(element() * self.observations_scale[i])

        observation_history = np.concatenate(
            [
                np.array(list(self.observation_histories[i]), dtype=np.float32).flatten()
                for i in range(len(self.observations_func))
            ],
        )
        return observation_history

    def base_ang_vel(self):
        return self.state["base_angular_vel"]

    def projected_gravity(self):
        qw, qx, qy, qz = self.state["base_orientation"]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def generated_commands(self):
        scaled_command = (self.command + 1) / 2 * (
            self.commands_ranges["upper"] - self.commands_ranges["lower"]
        ) + self.commands_ranges["lower"]
        scaled_command[np.abs(scaled_command) < self.commands_ranges["velocity_deadzone"]] = 0.0
        return scaled_command

    def joint_pos_rel(self):
        return self.state["qpos"] - self.default_joint_pos

    def joint_vel_rel(self):
        return self.state["qvel"] - self.default_joint_vel

    def last_action(self):
        return self.actions

    def cos_phase(self):
        count = self.counter * self.control_dt
        phase = count % self.period / self.period
        return np.cos(2 * np.pi * phase)

    def sin_phase(self):
        count = self.counter * self.control_dt
        phase = count % self.period / self.period
        return np.sin(2 * np.pi * phase)


class ActionHandler:
    def __init__(self, action_scale, default_joint_pos):
        self.action_scale = action_scale
        self.default_joint_pos = default_joint_pos

    def get_scaled_action(self, action) -> np.ndarray:
        return self.action_scale * action + self.default_joint_pos


class RLPolicy(Policy):
    def __init__(self, robot: Robot, policy_dir: Path, log_data: bool = False):
        self._load_config(policy_dir)
        self._get_policy_path(policy_dir)
        self._get_joint_config(robot)

        self.log_data = log_data
        self.control_dt = self.config["control_dt"]

        enabled_default_joint_pos = self.default_joint_pos[self.enabled_joints_idx]
        history_length = self.config["history_length"]
        action_scale = self.config["action_scale"]

        command_ranges = {
            "lower": np.array([cmd_range[0] for cmd_range in self.config["command_ranges"].values()]),
            "upper": np.array([cmd_range[1] for cmd_range in self.config["command_ranges"].values()]),
            "velocity_deadzone": self.config["velocity_deadzone"],
        }
        observations_func = [obs["name"] for obs in self.config["observations"]]
        observations_scale = [obs.get("scale") or 1 for obs in self.config["observations"]]

        if self.policy_path.suffix == ".pt":
            self.policy = InferenceHandlerTorch(policy_path=(self.policy_path))
        elif self.policy_path.suffix == ".onnx":
            self.policy = InferenceHandlerONNX(policy_path=str(self.policy_path))
        else:
            msg = f"Unsupported file extension for policy_path: {self.policy_path}. Only .pt and .onnx are supported."
            raise ValueError(msg)
        self.observation_handler = ObservationHandler(
            observations_func,
            observations_scale,
            history_length,
            enabled_default_joint_pos,
            command_ranges,
        )
        self.action_handler = ActionHandler(action_scale, enabled_default_joint_pos)
        self.actions = np.zeros_like(enabled_default_joint_pos)

        if self.log_data:
            self.logger = RLLogger()

    def step(
        self, state: dict, command: np.ndarray | None = None
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state["qpos"] = state["qpos"][self.enabled_joints_idx]
        state["qvel"] = state["qvel"][self.enabled_joints_idx]

        q_ref = self._policy_step(state, command)
        q_whole = self.default_joint_pos.copy()
        q_whole[self.enabled_joints_idx] = q_ref
        dq_ref = np.zeros_like(q_whole)
        return self.control_dt, q_whole, dq_ref, self.kps, self.kds

    def save_data(self, log_dir=None):
        if log_dir is not None:
            self.logger.save_data(log_dir=log_dir)

    def _load_config(self, policy_dir: Path):
        config_path = policy_dir / "env.yaml"
        if not config_path.exists():
            err_msg = f'No policy config `env.yaml` found in directory "{policy_dir}"'
            raise ConfigError(err_msg)

        with config_path.open() as f:
            self.config = yaml.safe_load(f)

    def _get_policy_path(self, policy_dir: Path) -> None:
        policy_paths = [path for path in policy_dir.iterdir() if path.is_file() and path.suffix in (".pt", ".onnx")]
        if len(policy_paths) != 1:
            if len(policy_paths) == 0:
                err_msg = f'No policy file found in directory "{policy_dir}" (no `.pt` or `.onnx` file)'
            else:
                err_msg = f'Multiple policy file found in directory "{policy_dir}": {", ".join(map(str, policy_paths))}'
            raise ConfigError(err_msg) from None
        self.policy_path = policy_paths[0]

    def _get_joint_config(self, robot: Robot) -> None:
        robot_joints = robot.get_joint_names()
        num_joints = len(robot_joints)
        self.kps = np.empty(num_joints)
        self.kds = np.empty(num_joints)
        self.default_joint_pos = np.empty(num_joints)

        for joint_id, joint_name in enumerate(robot_joints):
            joint_config = [joint for joint in self.config["joints"] if joint["name"] == joint_name]
            if len(joint_config) != 1:
                if len(joint_config) == 0:
                    err_msg = f"Joint '{joint_name}' is not set up in the config file"
                else:
                    err_msg = f"Found multiple config for joint '{joint_name}' in the config file"
                raise ConfigError(err_msg)
            joint_config = joint_config[0]
            self.kps[joint_id] = joint_config["kp"]
            self.kds[joint_id] = joint_config["kd"]
            self.default_joint_pos[joint_id] = joint_config["default_joint_pos"]

        enabled_joints_idx = []
        for joint in self.config["joints"]:
            if not joint["enabled"]:
                continue
            if joint["name"] not in robot_joints:
                err_msg = f"Joint '{joint['name']}' is enabled, but cannot be found in the model"
                raise ConfigError(err_msg)

            joint_id = robot_joints.index(joint["name"])
            enabled_joints_idx.append(joint_id)
        self.enabled_joints_idx = np.array(enabled_joints_idx)

    def _policy_step(self, state: dict, command: np.ndarray | None) -> np.ndarray:
        observations = self.observation_handler.get_observations(state, self.actions, command)
        self.actions = self.policy(observations)

        if self.log_data:
            self.logger.record_metrics(observations=observations, actions=self.actions)

        return self.action_handler.get_scaled_action(self.actions)
