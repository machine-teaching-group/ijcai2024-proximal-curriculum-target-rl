from abstract_classes.AbstractTeacher import AbstractTeacherCallback
import numpy as np


class MinigridTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the MazeEnv environment.
        Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, type_env, target_type, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                 currot_perf_lb, currot_metric_eps, metrics=True, verbose=1):
        self.type_env = env.type_env
        norm_strategy = None
        self.TARGET_LOWER_CONTEXT_BOUNDS = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        self.TARGET_UPPER_CONTEXT_BOUNDS = np.array([1, 0, 0, 0, 1, 1, 1, 1])
        super(MinigridTeacher, self).__init__(env, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                                         currot_perf_lb, currot_metric_eps, norm_strategy, metrics, verbose)

        self.target_type = target_type

    @staticmethod
    def get_context_obs(curr_task, ind):
        s0, _ = curr_task.reset(seed=ind)
        return s0

    @staticmethod
    def set_context_bounds():
        LOWER_CONTEXT_BOUNDS = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        UPPER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        return LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS

    @staticmethod
    def take_step(curr_task, action):
        s1, r, done, info = curr_task.step(action)
        return s1, r, done, info

    @staticmethod
    def reset_task(curr_task, index):
        curr_task.reset()
        return

    def get_vmax(self):
       pass

    def target_sampler(self, n, rng=None):

        if rng is None:
            rng = np.random

        if self.target_type == "single-target":
            samples = []
            for i in range(n):
                samples.append(np.stack((1, 0, 0, 0, 1, 1, 1, 1), axis=-1))
            return np.array(samples)
        else:
            raise ValueError("Unknown target type")


    @staticmethod
    def check_horizon(step, done):
        if step == 2000:
            done = True
        return done
