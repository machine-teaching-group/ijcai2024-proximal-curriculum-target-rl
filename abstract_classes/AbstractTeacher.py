from abc import ABC, abstractmethod
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from tqdm import tqdm
import wandb
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from utils.currot_utils.wasserstein_interpolation import WassersteinInterpolation
from utils.currot_utils.util import RewardEstimatorGP
from utils.currot_utils.util import Buffer
from utils.currot_utils.buffers import WassersteinSuccessBuffer


class AbstractTeacherCallback(BaseCallback, ABC):
    """ Abstract class for the teacher callback inherits from stable-baselines3 BaseCallback.
        Template for defining a concrete teacher callback that can be used with sb3
        while training on any gym environment with a curriculum wrapper"""
    def __init__(self, env, cur, eps, n_steps, spdl_pthresh, Na, alpha, epsilon,
                 currot_perf_lb, currot_metric_eps, norm_strategy, metrics=True, verbose=1):
        super(AbstractTeacherCallback, self).__init__(verbose)

        self.env = env  # A gym env that is defined based on our AbstractGymWrapper class
        self.instances = self.env.collection_envs
        self.cur = cur
        self.eps = eps  # Noise added to the critic estimate
        self.n_steps = n_steps  # nb_steps for model update
        self.pos_steps = 0  # To measure the environment steps of the extra rollouts
        self.metrics = metrics
        self.type_env = self.env.type_env  # Type of env based on success and reward. Can be binary, non-binary, info based success
        self.nb_eval_rolls = 20  # Hyperparameter nb_rollouts to evaluate learner
        self.step_budget = np.inf  # Can be used to restrict budget of steps for PoS evaluation
        self.norm_strategy = norm_strategy

        # Spdl parameters
        self.Na = Na
        self.alpha = alpha
        self.epsilon = epsilon
        self.spdl_pthresh = spdl_pthresh  # performance threshold for SPDL

        # For PLR
        self.rollout_tasks = []

        # For currot
        if self.cur == "currot":

                self.threshold_reached = True
                perf_lb = currot_perf_lb
                init_samples = self.env.init_samples
                self.episodes_per_update = 50
                metric_eps = currot_metric_eps
                LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS = self.set_context_bounds()
                self.context_bounds = (LOWER_CONTEXT_BOUNDS.copy(), UPPER_CONTEXT_BOUNDS.copy())
                self.gpmodel = RewardEstimatorGP()
                self.success_buffer = WassersteinSuccessBuffer(perf_lb, init_samples.shape[0], self.episodes_per_update, metric_eps,
                                                               context_bounds=self.context_bounds, max_reuse=1)
                self.fail_context_buffer = []
                self.fail_return_buffer = []
                callback = None
                self.wasserstein_interp = WassersteinInterpolation(init_samples, self.target_sampler, perf_lb, epsilon, callback=callback)
                self.context_buffer = Buffer(2, self.episodes_per_update + 1, True)

    @abstractmethod
    def target_sampler(self, n, rng=None):
        pass

    @staticmethod
    @abstractmethod
    def set_context_bounds():
        pass

    @staticmethod
    @abstractmethod
    def get_context_obs(curr_task, index):
        pass

    @staticmethod
    @abstractmethod
    def take_step(curr_task, action):
        pass

    @staticmethod
    @abstractmethod
    def reset_task(curr_task, cur_id):
        pass

    @abstractmethod
    def get_vmax(self):
        pass

    @staticmethod
    @abstractmethod
    def check_horizon(cur_step, done):
        pass

    # Sampling function to generate an instance of the target space
    def get_expected_target_performance(self):

        p_target = []
        for _ in tqdm(range(self.env.num_of_target_samples)):

            target_context = self.env.generate_one_target_context(type=self.env.target_type)

            p_target.append(self.value_critic_with_context(target_context) *
                            (1 - self.value_critic_with_context(target_context)) *
                            self.env.sim_kernel_with_context(target_context))

        return np.array(p_target)

    def value_forward_pass(self, single_instance=None):
        """
        Do forward pass of the value network for all tasks and return value estimates
        :return: numpy array evals, the value prediction for each task
        """

        if single_instance is None:
            instances = self.instances
        else:
            instances = [self.instances[single_instance]]

        evals = []
        for index, curr_task in tqdm(enumerate(instances)):
            s0 = self.get_context_obs(curr_task, index)
            s_tensor, _ = self.model.policy.obs_to_tensor(s0)
            max_future_q = self.model.policy.predict_values(s_tensor)
            evals.append(max_future_q.cpu().detach().numpy()[0][0])
        evals = np.array(evals)
        return evals

    def value_forward_pass_with_context(self, context):

        context_state = self.env.get_context_state_obs(context)
        s_tensor, _ = self.model.policy.obs_to_tensor(context_state)
        max_future_q = self.model.policy.predict_values(s_tensor)
        return np.array(max_future_q.cpu().detach().numpy()[0][0])

    # Used for space variant

    def value_critic(self, single_instance=None):
        """
        Compute values for each task. Normalize values
        and clip them to range [Vmin, Vmax]
        """

        if single_instance is not None:
            evals = self.value_forward_pass(single_instance)
        else:
            evals = self.value_forward_pass()

        if self.norm_strategy == "adaptive":
            evals = (evals - np.min(evals)) / (np.max(evals) - np.min(evals))

        elif self.norm_strategy == "static":
            vm = self.get_vmax()
            evals = evals / vm

        # # Noise is added to the critic estimation. Used for ablation
        # if self.eps > 0:
        #     noise = np.random.uniform(-self.eps, self.eps, self.env.num_of_tasks)  # Add uniform noise to the evaluations
        #     evals = np.add(evals, noise)
        norm_val = np.clip(evals, a_min=0, a_max=1)  # Clipping between Vmin and Vmax

        return norm_val

    def value_critic_with_context(self, context):
        evals = self.value_forward_pass_with_context(context)

        if self.norm_strategy == "adaptive":
            evals = (evals - np.min(evals)) / (np.max(evals) - np.min(evals))

        elif self.norm_strategy == "static":
            vm = self.get_vmax()
            evals = evals / vm

        norm_val = np.clip(evals, a_min=0, a_max=1)
        return norm_val

    # Currot Update
    def Currot_update_distribution(self, contexts, returns):
        fail_contexts, fail_returns = self.success_buffer.update(contexts, returns,
                                                                 self.wasserstein_interp.target_sampler(
                                                                     self.wasserstein_interp.current_samples.shape[0]))

        if self.threshold_reached:
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.wasserstein_interp.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.wasserstein_interp.n_samples:]

        success_contexts, success_returns = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            train_contexts = success_contexts
            train_returns = success_returns
        else:
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)
        self.gpmodel.update_model(train_contexts, train_returns)

        if self.threshold_reached or self.gpmodel(self.wasserstein_interp.current_samples) >= self.wasserstein_interp.perf_lb:
            self.threshold_reached = True
            self.wasserstein_interp.update_distribution(self.gpmodel, self.success_buffer.read_update()) # The reward estimator model
        else:
            print("Not updating sampling distribution, as performance threshold not met: %.3e vs %.3e" % (
                self.gpmodel(self.wasserstein_interp.current_samples), self.wasserstein_interp.perf_lb))

    # SPDL Update
    @staticmethod
    def logsumexp(x, axis=None):
        x_max = np.max(x, axis=axis)
        return np.log(np.sum(np.exp(x - x_max), axis=axis)) + x_max

    @staticmethod
    def interp_ll(old_ll, target_ll, values, x):
        eta, alpha = x
        interp_ll = (target_ll + eta * values) * alpha + old_ll * (1 - alpha)
        interp_ll -= AbstractTeacherCallback.logsumexp(interp_ll)
        return interp_ll

    @staticmethod
    def expected_value(old_ll, target_ll, values, x):
        return np.sum(np.exp(AbstractTeacherCallback.interp_ll(old_ll, target_ll, values, x)) * values)

    @staticmethod
    def kl_divergence(old_ll, target_ll, ref_ll, values, x):
        interp_ll = AbstractTeacherCallback.interp_ll(old_ll, target_ll, values, x)
        return np.sum(np.exp(interp_ll) * (interp_ll - ref_ll))

    def spdl_update(self):
        """
        SPDL optimization step
        """
        evals = self.value_forward_pass()
        min_eval = np.min(evals)
        max_eval = np.max(evals)
        eval_range = max_eval - min_eval
        # We normalize the values to be between 0 and 1
        if self.type_env == "non-binary" or self.type_env == "info-based":
            print("Min: %.3e, Max: %.3e, Range: %.3e" % (min_eval, max_eval, eval_range))
            evals = (evals - min_eval) / eval_range

        # Compute ci
        if hasattr(self.env, "log_pv"):
            old_ll = self.env.log_pv
        else:
            old_ll = np.log(self.env.pv)

        # Define the spdl objective
        from functools import partial

        if self.type_env == "non-binary" or self.type_env == "info-based":
            delta = (self.spdl_pthresh - min_eval) / eval_range
        else:
            delta = self.spdl_pthresh
        print("Iteration Delta: %.3e" % delta)
        # Want a uniform distribution over the contexts
        target_ll = -np.log(old_ll.shape[0]) * np.ones(old_ll.shape[0])
        perf_con = NonlinearConstraint(partial(self.expected_value, old_ll, target_ll, evals), delta, np.inf)
        kl_con = NonlinearConstraint(partial(self.kl_divergence, old_ll, target_ll, old_ll, evals), -np.inf,
                                     self.epsilon)

        # If we are below the performance threshold, we optimize the performance in a first run
        avg_perf = np.sum(np.exp(old_ll) * evals)
        if avg_perf <= delta:
            neg_objective = partial(self.expected_value, old_ll, target_ll, evals)
            res = minimize(lambda x: -neg_objective(x), np.array([1., 1.]), method='trust-constr', jac="3-point",
                           constraints=[kl_con], options={'verbose': 1, "gtol": 1e-4, "xtol": 1e-6},
                           bounds=Bounds(1e-3 * np.ones(2), 1e4 * np.ones(2)))

            intermediate_ll = self.interp_ll(old_ll, target_ll, evals, res.x)
            x0 = res.x

            avg_perf = np.sum(np.exp(intermediate_ll) * evals)
            if res.success:
                # In this case we either set the optimized performance distribution as the new sampling distributoin
                if avg_perf < delta:
                    print("Optimized performance as performance constraint not met: %.3e vs %.3e" % (avg_perf, delta))
                    self.env.log_pv = intermediate_ll
                    self.env.pv = np.exp(intermediate_ll)
                    return
            else:
                print("Warning! Optimization not successful")
                return
            # Only if the optimization was successful and the optimized result fulfills the performance constraint
            # we continue with the optimization.
        else:
            intermediate_ll = old_ll
            x0 = np.array([1., 1.])

        # If we start above the performance threshold, we minimize the KL
        if avg_perf > delta:
            constraints = [perf_con, kl_con]
            objective = partial(self.kl_divergence, old_ll, target_ll, target_ll, evals)

            res = minimize(objective, x0, method='trust-constr', jac="3-point", constraints=constraints,
                           options={'verbose': 1, "gtol": 1e-8, "xtol": 1e-8}, #options={'verbose': 1, "gtol": 1e-4, "xtol": 1e-6},
                           bounds=Bounds(1e-4 * np.ones(2), 1e4 * np.ones(2)))

            if res.success:
                print("New Target KL-Divergence: %.3e" % res.fun)
                self.env.log_pv = self.interp_ll(old_ll, target_ll, evals, res.x)
            else:
                print("Warning! Optimization not successful!")
                self.env.log_pv = intermediate_ll
        else:
            raise RuntimeError("Should not happen!")

        self.env.pv = np.exp(self.env.log_pv)
        print("Expected Performance: %.3e" % self.expected_value(old_ll, target_ll, evals, res.x))

    def _on_training_start(self) -> None:
        print(f"Instantiate {self.cur} Teacher")
        # Domain knowledge assumed pos*=1
        # PosL is estimated by a value critic
        if self.cur == "procurl-val":
            self.env.posl = self.value_critic()

        elif self.cur == "proxcorl":
            self.env.posl = self.value_critic()
            self.env.c2ctest = self.get_expected_target_performance()

        elif self.cur == "currot":
            # Already initialized with init_samples
            pass

    def _on_rollout_start(self) -> None:
        self.rollout_tasks = []

    def _on_rollout_end(self) -> None:

        if self.cur == "plr":
            # These are the indices of the start of each task in the rollout
            task_starts = np.where(self.locals['rollout_buffer'].episode_starts == 1)[0]

            # if task_tasks does not contain 0 we add it
            if task_starts[0] != 0:
                task_starts = np.insert(task_starts, 0, 0)

            # If rollout_tasks has more elements than task_starts we remove the last task in rollout_tasks
            if len(self.rollout_tasks) > len(task_starts):
                self.rollout_tasks = self.rollout_tasks[:-1]

            # Compute the mean advantage of the rollout for each task in the rollout
            for i, task in enumerate(self.rollout_tasks):
                if i < len(task_starts) - 1:
                    # Update the environment p_scores if it is not nan
                    temp = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:task_starts[i+1] -1])
                    if not np.isnan(temp):
                        self.env.p_scores[task] = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:task_starts[i+1] -1])
                    else:
                        # Keep the old p_score if the new one is nan
                        #print("Warning: nan value in p_scores. Keeping old p_score.")
                        self.env.p_scores[task] = self.env.p_scores[task]
                else:
                    # Update the environment p_scores if it is not nan
                    temp = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:])
                    if not np.isnan(temp):
                        self.env.p_scores[task] = np.mean(self.locals['rollout_buffer'].advantages[task_starts[i]:])
                    else:
                        # Keep the old p_score if the new one is nan
                        #print("Warning: nan value in p_scores. Keeping old p_score.")
                        self.env.p_scores[task] = self.env.p_scores[task]

            # Check if the p_scores contain nan values and raise an error if so
            if np.isnan(self.env.p_scores).any():
                raise ValueError("p_scores contain nan values")

    def _on_step(self) -> bool:

        # Log metrics(total number of steps)
        if self.metrics:
            if self.n_calls % self.n_steps == 0:
                wandb.log({"env_n_calls": self.n_calls, "global_env_steps": self.n_calls + self.pos_steps})

        # If new task is selected or new rollout starts
        if self.locals["tb_log_name"] == "SAC":
            if self.env.num_steps == 0 or self.locals['num_collected_steps'] == 0:
                self.rollout_tasks.append(self.env.cur_id)
        else:
            if self.env.num_steps == 0 or self.locals['n_steps'] == 0:
                self.rollout_tasks.append(self.env.cur_id)

        # Add to context buffer for currot
        if self.cur == "currot":
            if self.locals["tb_log_name"] == "SAC":
                if self.locals['done'] == True:
                    self.context_buffer.update_buffer((self.env.currot_contx, self.locals["reward"][0]))
            else:
                if self.locals['dones'] == True:
                    self.context_buffer.update_buffer((self.env.currot_contx, self.locals["rewards"][0]))

        # Update the distribution for currot
        if self.cur == "currot":
            if len(self.context_buffer) >= self.episodes_per_update:
                contexts, returns = self.context_buffer.read_buffer()
                self.Currot_update_distribution(contexts=np.array(contexts), returns=np.array(returns))
                self.env.currot_current_tasks = self.wasserstein_interp.current_samples

        if self.n_calls % (self.n_steps) == 0:

            # Value critic evaluation
            if self.cur == "procurl-val":
                # Estimated posl from value critic
                self.env.posl = self.value_critic()

            elif self.cur == "proxcorl":
                self.env.posl = self.value_critic()
                self.env.c2ctest = self.get_expected_target_performance()

            # Spdl optimization step
            elif self.cur == "spdl":
                self.spdl_update()

        return True





