import numpy as np
import onnxruntime as ort
from collections import deque


class InferenceHandler:
    def __init__(self, policy_path):
        self.ort_sess = ort.InferenceSession(policy_path)

    def inference(self, observations):
        actions = self.ort_sess.run(None, {"obs": observations})[0]

        return actions.flatten()


class ObservationHandler:
    def __init__(self):
        self.observation_history = deque(maxlen=10)

    def get_observations(self, state):
        observation = np.concatenate(
            self._get_base_ang_vel(state),
            self._get_gravity_orientation(state[3:7]),
            self._get_command(),
            self._get_joint_pos(),
            self._get_joint_vel(),
            self._previous_action,
        )

        self.observation_history.append(observation)
        observation_history = np.array(self.observation_history)

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
        return state[7 : 12 + 7]

    def _get_joint_vel(self, state):
        return state[12 + 7 + 6 :]


class ActionHandler:
    def __init__(self, action_scale, default_joint_pos):
        self.action_scale = action_scale
        self.default_joint_pos = default_joint_pos

    def get_scaled_action(self, action):
        return self.action_scale * action + self.default_joint_pos


class RLPolicy:
    def __init__(self, policy_path):
        self.policy = InferenceHandler(policy_path=policy_path)

        self.observation_handler = ObservationHandler()
        self.action_handler = ActionHandler(0.5, np.array([0]))

    def step(self, state):
        observations = self.observation_handler.get_observations(state)
        actions = self.policy.inference(observations)
        scaled_actions = self.action_handler.get_scaled_action(actions)

        return scaled_actions


if __name__ == "__main__":
    path = "./controllers/rl_utils/policy.onnx"
    policy = RLPolicy(path)

    observation = np.ones((1, 93)).astype(np.float32)
    print(observation)
    action = policy.step(observation)
    print(action)

    from IPython import embed

    embed()
