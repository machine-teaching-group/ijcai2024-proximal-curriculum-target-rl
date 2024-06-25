import gymnasium as gym
import numpy as np
from gym import utils
from envs.sgr.viewer import Viewer


class MazeEnv(gym.core.Env, utils.EzPickle):

    def __init__(self, context=np.array([0., 0.])):
        """
        The maze has the following shape:

        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],

        """

        self.action_space = gym.spaces.Box(np.array([-1., -1.]), np.array([1., 1.]))

        self.observation_space = gym.spaces.Box(np.array([-9., -9.]), np.array([9., 9.]))

        # Contextual obs
        self.observation_space_context = gym.spaces.Box(
            np.array([-9., -9., -9., -9., 0.05]), np.array([9., 9., 9., 9., 18]))

        self._state = None
        self.context = context
        self.context_state = None
        self.max_step = 0.3

        self._viewer = Viewer(20, 20, background=(255, 255, 255))
        self.num_step = 0
        gym.core.Env.__init__(self)
        utils.EzPickle.__init__(**locals())

    @staticmethod
    def sample_initial_state(n=None):
        if n is None:
            return np.random.uniform(-7., -5., size=(2,))
        else:
            return np.random.uniform(-7., -5., size=(n, 2))

    def set_context(self, context):
        self.context = context

    def reset(self):
        self._state = self.sample_initial_state()
        self.num_step = 0
        self.context_state = np.concatenate((self._state, self.context), axis=0)
        info = {}
        return np.copy(self.context_state), info

    @staticmethod
    def _is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        if np.any(context < -7.) or np.any(context > 7.):
            return False
        # Check that the context is not within the inner rectangle (i.e. in [-5, 5] x [-5, 5])
        elif np.all(np.logical_and(-5. < context, context < 5.)):
            return False
        else:
            return True

    @staticmethod
    def _project_back(old_state, new_state):
        # Project back from the bounds
        new_state = np.clip(new_state, -7., 7.)

        # Project back from the inner circle
        if -5 < new_state[0] < 5 and -5 < new_state[1] < 5:
            new_state = np.where(np.logical_and(old_state <= -5, new_state > -5), -5, new_state)
            new_state = np.where(np.logical_and(old_state >= 5, new_state < 5), 5, new_state)

        return new_state

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = self.max_step * (action / max(1., np.linalg.norm(action)))
        new_state = self._project_back(self._state, self._state + action)

        done = False

        info = {"success": (np.linalg.norm(self.context[:2] - new_state) < self.context[2])}
        info["reward"] = 1. if info["success"] else 0.
        self._state = np.copy(new_state)
        self.num_step += 1
        if self.num_step == 200 or info["success"]:
            done = True

        new_state = np.concatenate((new_state, self.context), axis=0)

        truncated = None  # Used to migrate from gym to gymnasium

        return new_state, info["reward"], done, truncated, info

    def render(self, mode='human'):
        offset = 10

        outer_border_poly = [np.array([-9, 1]), np.array([9, 1]), np.array([9, -1]), np.array([-9, -1])]

        self._viewer.polygon(np.array([0, -8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([0, 8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([-8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.square(np.zeros(2) + offset, 0., 10, color=(0, 0, 0))

        self._viewer.circle(self._state + offset, 0.6, color=(0, 0, 0))
        self._viewer.display(1, path="maze.bmp")


if __name__ == '__main__':

    env = MazeEnv(context=np.array([6.9,6.9,0.05]))
    print(env._is_feasible(env.context))