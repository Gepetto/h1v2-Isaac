from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from biped_deploy.utils.rl_logger import RLLogger


class InferenceHandlerONNX:
    def __init__(self, policy_path):
        self.ort_sess = ort.InferenceSession(policy_path)
        self.input_name = self.ort_sess.get_inputs()[0].name

    def __call__(self, observations):
        observations_unsqueezed = np.expand_dims(observations, axis=0)
        actions = self.ort_sess.run(None, {self.input_name: observations_unsqueezed})[0]

        return actions.flatten()


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


class RLPolicy:
    def __init__(self, policy_path, config, log_data=False):
        default_joint_pos = np.array([joint["default_joint_pos"] for joint in config["joints"] if joint["enabled"]])
        history_length = config["history_length"]
        action_scale = config["action_scale"]

        command_ranges = {
            "lower": np.array([cmd_range[0] for cmd_range in config["command_ranges"].values()]),
            "upper": np.array([cmd_range[1] for cmd_range in config["command_ranges"].values()]),
            "velocity_deadzone": config["velocity_deadzone"],
        }
        observations_func = [obs["name"] for obs in config["observations"]]
        observations_scale = [obs.get("scale") or 1 for obs in config["observations"]]

        self.log_data = log_data
        if self.log_data:
            self.logger = RLLogger()

        if policy_path.endswith(".pt"):
            self.policy = InferenceHandlerTorch(policy_path=policy_path)
        elif policy_path.endswith(".onnx"):
            self.policy = InferenceHandlerONNX(policy_path=policy_path)
        else:
            raise ValueError(
                f"Unsupported file extension for policy_path: {policy_path}. Only .pt and .onnx are supported."
            )
        self.observation_handler = ObservationHandler(
            observations_func,
            observations_scale,
            history_length,
            default_joint_pos,
            command_ranges,
        )
        self.action_handler = ActionHandler(action_scale, default_joint_pos)

        self.actions = np.zeros_like(default_joint_pos)

    def step(self, state, command=None):
        observations = self.observation_handler.get_observations(state, self.actions, command)
        self.actions = self.policy(observations)

        if self.log_data:
            self.logger.record_metrics(observations=observations, actions=self.actions)

        return self.action_handler.get_scaled_action(self.actions)

    def save_data(self, log_dir=None):
        if log_dir is not None:
            self.logger.save_data(log_dir=log_dir)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    policy_path = str(Path(__file__).parent / "config" / "agent_model.onnx")
    policy = RLPolicy(policy_path, config["rl"])

    state = np.zeros(12 * 2 + 7 + 6)
    print(policy.step(state))
