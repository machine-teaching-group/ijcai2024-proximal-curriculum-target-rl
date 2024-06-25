from abstract_classes.AbstractTeacher import AbstractTeacherCallback
import numpy as np


class SgrTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the MazeEnv environment.
        Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, type_env, target_type, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                 currot_perf_lb, currot_metric_eps, metrics=True, verbose=1):
        self.type_env = env.type_env
        norm_strategy = None
        self.TARGET_LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
        self.TARGET_UPPER_CONTEXT_BOUNDS = np.array([9., 9., 0.05])
        super(SgrTeacher, self).__init__(env, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                                         currot_perf_lb, currot_metric_eps, norm_strategy, metrics, verbose)

        self.type = target_type

    @staticmethod
    def get_context_obs(curr_task, ind):
        s0 = curr_task.context_state
        return s0

    @staticmethod
    def set_context_bounds():
        LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
        UPPER_CONTEXT_BOUNDS = np.array([9., 9., 18.])
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

        if self.type == "single-mode-gaussian":

           s0 = rng.uniform(low=-9, high=9, size=(n, 1))
           s1 = rng.uniform(low=-9, high=9, size=(n, 1))
           s2 = rng.normal(0.06, 0.01, size=(n, 1))
           # Clip s3 to be larger than 0.05
           s2 = np.clip(s2, a_min=0.05, a_max=None)
           sample = np.concatenate([s0, s1, s2], axis=1)
           return sample

        elif self.type == "double-mode-gaussian":
            # For maze double mode gaussian

            decisions = rng.randint(0, 2, size=n)
            s0 = rng.uniform(low=-9, high=9, size=(n, 1))
            s1 = rng.uniform(low=-9, high=9, size=(n, 1))
            s2 = rng.normal(0.06, 0.01, size=(n, 1))
            # Clip s3 to be larger than 0.05
            s2 = np.clip(s2, a_min=0.05, a_max=None)
            sample1 = np.concatenate([s0, s1, s2], axis=1)

            s0 = rng.uniform(low=-9, high=9, size=(n, 1))
            s1 = rng.uniform(low=-9, high=9, size=(n, 1))
            s2 = rng.normal(0.12, 0.01, size=(n, 1))
            # Clip s3 to be larger than 0.05
            s2 = np.clip(s2, a_min=0.05, a_max=None)
            sample2 = np.concatenate([s0, s1, s2], axis=1)

            ret = decisions[:, None] * sample1 + (1 - decisions)[:, None] * sample2
            return ret

        elif self.type == "single-plane":

            return rng.uniform(self.TARGET_LOWER_CONTEXT_BOUNDS, self.TARGET_UPPER_CONTEXT_BOUNDS, size=(n, 3))

        elif self.type == "single-task":

            samples = []
            for i in range(n):
                samples.append(np.stack((6.0, 6.0, 0.05), axis=-1))
            return np.array(samples)
        else:
            raise ValueError("Unknown target type")


    @staticmethod
    def check_horizon(step, done):
        if step == 200:
            done = True
        return done
