import numpy as np
import gymnasium as gym
from envs.pointmass.viewer import Viewer


class BinaryContextualPointMass(gym.Env):
    """
    Based on the Original Environment from "https://github.com/psclklnk/spdl.
    Changed the reward function to be sparse and binary.
    """

    def __init__(self, context=np.array([0.0, 2.0, 2.0])):

        self.action_space = gym.spaces.Box(
            np.array([-10.0, -10.0]), np.array([10.0, 10.0])
        )
        self.observation_space = gym.spaces.Box(
            np.array([-4.0, -np.inf, -4.0, -np.inf]),
            np.array([4.0, np.inf, 4.0, np.inf]),
        )
        # Contextual obs
        self.observation_space_context = gym.spaces.Box(
            np.array([-4.0, -np.inf, -4.0, -np.inf, -4, 0.5, 0]),
            np.array([4.0, np.inf, 4.0, np.inf, 4, 8, 4]),
        )

        self.state = None
        self.context_state = None
        self.goal_state = np.array([0.0, 0.0, -3.0, 0.0])
        self.context = context
        self._dt = 0.01
        self._viewer = Viewer(8, 8, background=(255, 255, 255))
        self.num_step = 0

    def get_context(self):
        return self.context

    def set_context(self, context):
        self.context = context

    def reset(self):
        self.state = np.array([0.0, 0.0, 3.0, 0.0])
        self.num_step = 0
        self.context_state = np.concatenate((self.state, self.context), axis=0)
        info = {}
        return np.copy(self.context_state), info

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        friction_param = self.context[2]
        state_der[1::2] = (
            1.5 * action
            - friction_param * state[1::2]
            + np.random.normal(0, 0.05, (2,))
        )
        # Do the update
        new_state = np.clip(
            state + self._dt * state_der,
            self.observation_space.low,
            self.observation_space.high,
        )

        done = False
        if state[2] >= 0 > new_state[2] or state[2] <= 0 < new_state[2]:
            alpha = (0.0 - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - self.context[0]) > 0.5 * self.context[1]:
                new_state = np.array([x_crit, 0.0, 0.0, 0.0])
                done = True

        return new_state, done

    def step(self, action):

        if self.state is None:
            raise RuntimeError(
                "State is None! Be sure to reset the environment before using it"
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        new_state = self.state
        done = False
        for i in range(0, 10):
            new_state, done = self._step_internal(new_state, action)
            if done:
                break

        self.state = np.copy(new_state)
        self.num_step += 1

        if self.num_step == 100:
            done = True

        info = {
            "success": np.linalg.norm(self.goal_state[0::2] - new_state[0::2])
            < 0.30
        }

        # Rewarded with 1 when point-mass reaches close enough to target, otherwise 0 reward.
        # Since envs with sparse rewards are harder, target is less than 0.30 compared to 0.25 of the original.
        if np.linalg.norm(self.goal_state[0::2] - new_state[0::2]) < 0.30:
            reward = 1
            done = True
        else:
            reward = 0

        truncated = None # To be compatible with gymnasium

        return (
            np.concatenate((new_state, self.context), axis=0),
            reward,
            done,
            truncated,
            info,
        )

    def render(self, mode="human"):
        pos = self.context[0] + 4.0
        width = self.context[1]
        self._viewer.line(
            np.array([0.0, 4.0]),
            np.array([np.clip(pos - 0.5 * width, 0.0, 8.0), 4.0]),
            color=(0, 0, 0),
            width=0.2,
        )
        self._viewer.line(
            np.array([np.clip(pos + 0.5 * width, 0.0, 8,), 4.0]),
            np.array([8.0, 4.0]),
            color=(0, 0, 0),
            width=0.2,
        )

        self._viewer.line(
            np.array([3.9, 0.9]),
            np.array([4.1, 1.1]),
            color=(255, 0, 0),
            width=0.1,
        )
        self._viewer.line(
            np.array([4.1, 0.9]),
            np.array([3.9, 1.1]),
            color=(255, 0, 0),
            width=0.1,
        )

        self._viewer.circle(
            self.state[0::2] + np.array([4.0, 4.0]), 0.1, color=(0, 0, 0)
        )
        self._viewer.display(5)
