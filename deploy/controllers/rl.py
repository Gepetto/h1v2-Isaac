import numpy as np
import onnxruntime as ort
from collections import deque


class InferenceHandler:
    def __init__(self, policy_path):
        self.ort_sess = ort.InferenceSession(policy_path)

    def inference(self, observations):
        observations_unsqueezed = np.expand_dims(observations, axis=0)
        actions = self.ort_sess.run(None, {"input": observations_unsqueezed})[0]

        return actions.flatten()


class ObservationHandler:
    def __init__(self, history_length, default_joint_pos):
        self.observation_history = deque(maxlen=history_length)
        self.default_joint_pos = default_joint_pos

    def get_observations(self, state, actions):
        observation = np.concatenate(
            [
                self._get_base_ang_vel(state),
                self._get_gravity_orientation(state[3:7]),
                self._get_command(),
                self._get_joint_pos(state),
                self._get_joint_vel(state),
                actions,
            ]
        )

        if len(self.observation_history) == 0:
            for _ in range(self.observation_history.maxlen):
                self.observation_history.append(observation)
        else:
            self.observation_history.append(observation)

        observation_history = np.array(
            self.observation_history, dtype=np.float32
        ).flatten()

        return observation_history

    def _get_base_ang_vel(self, state):
        return state[12 + 7 + 3 : 12 + 7 + 3 + 3]

    def _get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def _get_command(self):
        return np.zeros(3)

    def _get_joint_pos(self, state):
        return state[7 : 12 + 7] - self.default_joint_pos

    def _get_joint_vel(self, state):
        return state[12 + 7 + 6 :]


class ActionHandler:
    def __init__(self, action_scale, default_joint_pos):
        self.action_scale = action_scale
        self.default_joint_pos = default_joint_pos

    def get_scaled_action(self, action) -> np.ndarray:
        return self.action_scale * action + self.default_joint_pos


class RLPolicy:
    def __init__(self, policy_path):
        self._set_config()

        self.policy = InferenceHandler(policy_path=policy_path)

        self.observation_handler = ObservationHandler(self.history_length, self.default_joint_pos)
        self.action_handler = ActionHandler(self.action_scale, self.default_joint_pos)

        self.actions = np.zeros_like(self.default_joint_pos)

    def _set_config(self):
        self.history_length = 5

        self.action_scale = 0.5
        self.default_joint_pos = np.array(
            [0, -0.16, 0.0, 0.36, -0.2, 0.0, 0, -0.16, 0.0, 0.36, -0.2, 0.0],
        )

    def step(self, state):
        observations = self.observation_handler.get_observations(state, self.actions)
        self.actions = self.policy.inference(observations)
        scaled_actions = self.action_handler.get_scaled_action(self.actions)

        return scaled_actions


if __name__ == "__main__":
    path = "/home/cperrot/h1v2-Isaac/deploy/config/agent_model.onnx"
    policy = RLPolicy(path)

    state = np.zeros(12 * 2 + 7 + 6)
    print(policy.step(state))
