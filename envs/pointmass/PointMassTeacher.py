from abstract_classes.AbstractTeacher import AbstractTeacherCallback
import numpy as np


class PointMassTeacher(AbstractTeacherCallback):
    """Concrete teacher callback class for the PointMass environment.
        Inherits from our abstract teacher class so that different curriculum strategies can be used"""
    def __init__(self, env, type_env, target_type, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                 currot_perf_lb, currot_metric_eps, metrics=True, verbose=1):
        self.type_env = type_env  # Point-mass can be binary and non-binary
        self.TARGET_MEANS = np.array([[3.9, 0.5], [-3.9, 0.5]])
        self.TARGET_VARIANCES = np.array([np.diag([1e-4, 1e-4]), np.diag([1e-4, 1e-4])])  # standard deviation 0.05
        self.TARGET_VARIANCES = np.array(
            [np.diag([0.0025, 0.0025]), np.diag([0.0025, 0.0025])])  # standard deviation 0.05
        if self.type_env == "non-binary":
            norm_strategy = "adaptive"
        else:
            norm_strategy = None
        super(PointMassTeacher, self).__init__(env, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                                               currot_perf_lb, currot_metric_eps, norm_strategy, metrics, verbose)

        self.type = target_type

    @staticmethod
    def get_context_obs(curr_task, index):
        s0 = curr_task.context_state
        return s0

    @staticmethod
    def set_context_bounds():
        LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5, 0.])
        UPPER_CONTEXT_BOUNDS = np.array([4., 8., 4.])
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

            s0 = rng.multivariate_normal(self.TARGET_MEANS[0], self.TARGET_VARIANCES[0], size=n)
            friction_context = rng.uniform(0, 4, size=(n, 1))
            s0 = np.concatenate([s0, friction_context], axis=1)
            return s0

        elif self.type == "double-mode-gaussian":

            decisions = rng.randint(0, 2, size=n)
            s0 = rng.multivariate_normal(self.TARGET_MEANS[0], self.TARGET_VARIANCES[0], size=n)
            friction_context = rng.uniform(0, 4, size=(n, 1))
            s0 = np.concatenate([s0, friction_context], axis=1)

            s1 = rng.multivariate_normal(self.TARGET_MEANS[1], self.TARGET_VARIANCES[1], size=n)
            friction_context = rng.uniform(0, 4, size=(n, 1))
            s1 = np.concatenate([s1, friction_context], axis=1)

            ret = decisions[:, None] * s0 + (1 - decisions)[:, None] * s1
            return ret

        elif self.type == "single-task":

            context1 = 0.9
            context2 = 0.5
            context3 = 3.5
            target = np.array([context1, context2, context3])

            # replicate the target n times
            targets = np.tile(target, (n, 1))

            return targets

        else:
            raise ValueError("Unknown target type")

    @staticmethod
    def check_horizon(step, done):
        if step == 100:
            done = True
        return done
