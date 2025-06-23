from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml


class InferenceHandler:
    def __init__(self, policy_path):
        self.ort_sess = ort.InferenceSession(policy_path)

    def inference(self, observations):
        observations_unsqueezed = np.expand_dims(observations, axis=0)
        actions = self.ort_sess.run(None, {"input": observations_unsqueezed})[0]

        return actions.flatten()


class ObservationHandler:
    def __init__(
        self, history_length, default_joint_pos, commands_ranges, default_joint_vel=None, /, queue=None
    ):
        self.history_length = history_length
        self.default_joint_pos = default_joint_pos
        self.commands_ranges = commands_ranges
        self.default_joint_vel = (
            default_joint_vel
            if default_joint_vel is not None
            else np.zeros_like(default_joint_pos)
        )
        self.observation_histories = {}
        self.queue = queue
        self.command = np.array([0.0, 0.0, 0.0])

    def get_observations(self, state, actions):
        observation_elements = [
            self._get_base_ang_vel(state),
            self._get_gravity_orientation(state),
            self._get_command(),
            self._get_joint_pos(state),
            self._get_joint_vel(state),
            actions,
        ]

        for i, element in enumerate(observation_elements):
            if i not in self.observation_histories:
                self.observation_histories[i] = deque(maxlen=self.history_length)
                self.observation_histories[i].extend([element] * self.history_length)
            else:
                self.observation_histories[i].append(element)

        observation_history = np.concatenate(
            [
                np.array(
                    list(self.observation_histories[i]), dtype=np.float32
                ).flatten()
                for i in range(len(observation_elements))
            ],
        )
        return observation_history

    def _get_base_ang_vel(self, state):
        return state["base_angular_vel"]

    def _get_gravity_orientation(self, state):
        qw, qx, qy, qz = state["base_orientation"]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def _get_command(self):
        if self.queue is not None:
            while not self.queue.empty():
                self.command += self.queue.get()
                self.command = np.clip(self.command, self.commands_ranges["lower"], self.commands_ranges["upper"])
        return self.command

    def _get_joint_pos(self, state):
        return state["q_pos"] - self.default_joint_pos

    def _get_joint_vel(self, state):
        return state["q_vel"] - self.default_joint_vel


class ActionHandler:
    def __init__(self, action_scale, default_joint_pos):
        self.action_scale = action_scale
        self.default_joint_pos = default_joint_pos

    def get_scaled_action(self, action) -> np.ndarray:
        return self.action_scale * action + self.default_joint_pos


class RLPolicy:
    def __init__(self, policy_path, config, queue=None):
        default_joint_pos = np.array(
            [x for x in config["scene"]["robot"]["init_state"]["joint_pos"].values()]
        )
        history_length = config["observations"]["policy"]["history_length"]
        action_scale = config["actions"]["joint_pos"]["scale"]
        commands_ranges = {k: v for k, v in config["commands"]["base_velocity"]["ranges"].items() if v is not None}
        commands_ranges = {'lower': np.array([commands_ranges[key][0] for key in commands_ranges.keys()]), 
                           'upper': np.array([commands_ranges[key][1] for key in commands_ranges.keys()])}
        
        self.policy = InferenceHandler(policy_path=policy_path)
        self.observation_handler = ObservationHandler(
            history_length, default_joint_pos, commands_ranges, queue=queue
        )
        self.action_handler = ActionHandler(action_scale, default_joint_pos)

        self.actions = np.zeros_like(default_joint_pos)

    def step(self, state):
        observations = self.observation_handler.get_observations(state, self.actions)
        self.actions = self.policy.inference(observations)
        return self.action_handler.get_scaled_action(self.actions)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with config_path.open() as file:
        config = yaml.safe_load(file)

    policy_path = str(Path(__file__).parent / "config" / "agent_model.onnx")
    policy = RLPolicy(policy_path, config["rl"])

    state = np.zeros(12 * 2 + 7 + 6)
    print(policy.step(state))
