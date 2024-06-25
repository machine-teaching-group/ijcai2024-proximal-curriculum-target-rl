from gymnasium.core import ObservationWrapper
from gymnasium import spaces
import numpy as np
from functools import reduce
import operator

class FlatObsWrapper(ObservationWrapper):

    def __init__(self, env, context):
        super().__init__(env)

        self.context = context

        imgSpace = env.observation_space.spaces["image"]
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(imgSize+1+len(self.context),),
            dtype="uint8",
        )

    def observation(self, obs):
        image = obs["image"]
        flat_img = image.flatten()
        dir = np.array([obs["direction"]])

        # Concatenate the context with the image and direction
        obs = np.concatenate((flat_img, dir, self.context))

        return obs

