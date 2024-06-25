from abc import ABC, abstractmethod
import gymnasium as gym
import numpy as np
import random
from typing import Tuple, NoReturn


class AbstractCurriculumGymWrapper(gym.Env, ABC):
    """ Abstract class for a curriculum OpenAI Gym wrapper.
        Provides the template on how the concrete Gym environments
        should be created in order to run curriculum experiments.
        Any environment can be derived from this abstract class."""
    def __init__(self, cur, env_type, beta, target_type, path, beta_plr, rho_plr):

        self.cur = cur
        self.task_id, self.curr_task, self.cur_id, self.empty_env = None, None, None, None
        self.collection_envs, self.contexts, self.target_tasks, self.target_contexts = [], [], [], []
        self.ep = 0  # To log episodes
        self.target_type = target_type

        self.load_envs(path, env_type)
        LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS = self.set_context_bounds()
        context_size = len(LOWER_CONTEXT_BOUNDS)
        self.num_steps = 0
        self.num_of_tasks = len(self.collection_envs)

        # Random choice of the first task from collection_envs
        self.curr_task = np.random.choice(self.collection_envs)

        # Curriculum Multiple properties
        self.post = np.ones(self.num_of_tasks)  # POSTeacher=1
        self.posl = np.zeros(self.num_of_tasks)  # PoSLearner=0 and will be updated by the teacher callback
        self.posL_estim = np.zeros(self.num_of_tasks)  # PoSL estimation, used for updating PosL using training rollouts

        self.beta = beta
        self.iid_trigger = True

        # SPDL parameters
        self.pv = np.divide(np.ones(self.num_of_tasks), self.num_of_tasks)  # Initial uniform task distribution

        # PLR parameters
        self.binomial_p = 0.0
        self.seen = []
        self.unseen = list(range(self.num_of_tasks))
        self.p_scores = np.zeros(self.num_of_tasks)
        self.global_timestamps = np.zeros(self.num_of_tasks)
        self.rho_plr = rho_plr # Hyperparameter for P_replay
        self.beta_plr = beta_plr # Hyperparameter for P_replay

        # Target
        self.num_of_target_samples = 400#100 #400
        print("Number of target samples: ", self.num_of_target_samples)
        self.kernel = np.zeros(self.num_of_tasks)
        self.pstar_target = 1
        self.exp_target = np.zeros(self.num_of_tasks)
        # Create a list of the number of tasks
        self.cur_distribution = list(range(self.num_of_tasks))
        self.c2ctest = np.zeros((self.num_of_target_samples, self.num_of_tasks))
        self.exp_perf = 0

        # For currot
        self.init_samples = np.random.uniform(LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS, size=(500, context_size))
        self.target_contexts = np.array(self.target_contexts)
        self.currot_current_tasks = self.init_samples
        self.currot_contx = None
        self.context_bounds = (LOWER_CONTEXT_BOUNDS.copy(), UPPER_CONTEXT_BOUNDS.copy())
        self.sampler = UniformSampler(context_bounds=self.context_bounds)

        # Used for SPACE. It is used and updated by the teacher
        self.cur_set = [self.collection_envs[0]]
        self.cur_set_ids = [0]
        self.indices = [0]
        self.instance_set_size = 1 / len(self.collection_envs)
        self.space_id = 0
        self.dt = np.zeros(self.num_of_tasks)  # Difference in Values, i.e V_{t} - V_{t-1}


    @abstractmethod
    def load_envs(self, path, env_type):
        pass

    @staticmethod
    @abstractmethod
    def set_context_bounds():
        pass

    @staticmethod
    @abstractmethod
    def read_csv_tasks(path):
        pass

    @abstractmethod
    def generate_one_target_context(self, type):
        pass

    @abstractmethod
    def step(self, action):
        """
        Define an example step function of the environment
        """
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        obs, reward, done, info = self.curr_task.step(action=action)
        return obs, reward, done, info

    @abstractmethod
    def reset(self):
        """
        A reset function example of the environment
        """
        self.select_next_task()
        return self.curr_task.reset()  # Returns reset observation of the newly selected task

    @staticmethod
    def _sample_fast(arr, prob):
        return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

    # Sample a state from the target distribution
    def sample_target_state(self, nb):
        target_samples = []
        for _ in range(nb):
            target_samples.append(self._sample_fast(self.target_tasks, np.ones(len(self.target_tasks)) / len(self.target_tasks)))
        return target_samples

    @abstractmethod
    def set_env_with_context(self, context, seed):
        pass

    def select_next_task(self):
        """
        Selects next tasks based on the specified curriculum strategy
        """
        if self.cur == "proxcorl":

            self.cur_id = self.pick_curr_id_proxcorl()
            self.curr_task = self.collection_envs[self.cur_id]
            self.set_env_with_context(None, self.cur_id)

        elif self.cur == "target":

            self.target_contx = self.generate_one_target_context(self.target_type)
            self.set_env_with_context(self.target_contx, None)

        elif self.cur == "procurl-val":

            self.cur_id = self.pick_curr_id()
            self.curr_task = self.collection_envs[self.cur_id]
            self.set_env_with_context(None, self.cur_id)

        elif self.cur == "iid":

            self.cur_id = self.pick_random_id()
            self.curr_task = self.collection_envs[self.cur_id]
            self.set_env_with_context(None, self.cur_id)

        elif self.cur == "spdl":

            self.cur_id = self.pick_spdl()
            self.curr_task = self.collection_envs[self.cur_id]
            self.set_env_with_context(None, self.cur_id)
            self.curr_task.reset()

        elif self.cur == "currot":

            self.currot_contx = self.currot_sample()
            self.set_env_with_context(self.currot_contx, None)

        elif self.cur == "plr":

            self.cur_id = self.pick_plr()
            self.curr_task = self.collection_envs[self.cur_id]
            self.set_env_with_context(None, self.cur_id)

        # For gradient method
        elif self.cur == "gradient":

            random_idx = np.random.choice(self.inter_context.shape[0])
            self.grad_contx = self.inter_context[random_idx]
            self.set_env_with_context(self.grad_contx, None)

        else:

            raise ValueError("Invalid curriculum strategy")
        self.ep += 1  # A new episode

        return

    # Only used for metrics
    def get_contx_metrics(self):
        """
        Selects next tasks based on the specified curriculum strategy
        """
        if self.cur == "proxcorl":
            cur_id = self.pick_curr_id_proxcorl()
            contx = self.contexts[cur_id]
        elif self.cur == "target":
            contx = self.generate_one_target_context(self.target_type)
        elif self.cur == "procurl-val":
            cur_id = self.pick_curr_id()
            contx = self.contexts[cur_id]
        elif self.cur == "iid":
            cur_id = self.pick_random_id()
            contx = self.contexts[cur_id]
        elif self.cur == "spdl":
            cur_id = self.pick_spdl()
            contx = self.contexts[cur_id]
        elif self.cur == "currot":
            contx = self.currot_sample()
        elif self.cur == "plr":
            cur_id = self.pick_plr()
            contx = self.contexts[cur_id]
        else:
            raise ValueError("Invalid curriculum strategy")
        return contx

    def render(self, mode="human"):
        pass

    # Ways of selecting next tasks. Same across different environments
    # For gradient method
    def set_context_dist(self, context_dist):
        self.inter_context = context_dist

    def pick_random_id(self):
        return random.randint(0, self.num_of_tasks - 1)

    def currot_sample(self):
        sample = self.sampler.select(self.currot_current_tasks)
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])

    def pick_curr_id(self):
        """
        Select next task based on a soft selection from equation PoSL*(PosT-PoSL)
        """
        return random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=np.exp(self.beta * self.posl * (self.post - self.posl)), k=1)[0]

    def pick_curr_id_proxcorl(self):
        """
        Select next task based on a soft selection from target selection
        """

        perf_gradient = self.posl * (self.post - self.posl)
        matx = perf_gradient * self.c2ctest
        # Take the maximum weights per task c
        max_weights = np.max(matx, axis=0)
        id = random.choices(population=np.arange(0, self.num_of_tasks), weights=np.exp(self.beta * max_weights), k=1)[0]
        return id

    def pick_spdl(self):

        task = random.choices(population=np.arange(0, self.num_of_tasks),
                              weights=self.pv, k=1)[0]
        return task

    def pick_plr(self):

        # Anneal the bernoulli probability
        self.binomial_p = len(self.seen) / self.num_of_tasks

        # Sample replay decision from bernoulli distribution
        d = np.random.binomial(1, self.binomial_p)

        # Sample unseen task
        if d == 0 and self.unseen != []:

            # Sample uniform from list self.unseen
            task_id = random.choice(self.unseen)
            # Update seen and unseen
            self.seen.append(task_id)
            self.unseen.remove(task_id)
        else:
            # Update the staleness of the tasks
            self.update_staleness()
            self.update_p_scores()
            p_replay = (1-self.rho_plr) * self.P_s + self.rho_plr * self.P_c
            # Sample from seen tasks with prob Preplay
            task_id = random.choices(population=self.seen, weights=p_replay, k=1)[0]

        # Update the global timestamps for the task with episode number
        self.global_timestamps[task_id] = self.ep + 1
        return task_id

    def update_staleness(self):
        # Update the staleness of the seen tasks in self.seen
        seen_timestamps = self.global_timestamps[self.seen]
        self.P_c = (self.ep + 1 - seen_timestamps)/np.sum(self.ep + 1 - seen_timestamps, axis=0)
        return

    def update_p_scores(self):

        # Get the rank of each entry in self.scores
        seen_scores = self.p_scores[self.seen]
        temp = np.flip(np.argsort(seen_scores))
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(seen_scores)) + 1

        # Compute the weights and normalize
        weights = 1 / rank ** (1. / self.beta_plr)
        z = np.sum(weights)
        weights /= z
        self.P_s = weights
        return

    # Compute KL divergence between two distributions
    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    # Space functions
    def get_instance_set(self):
        return self.indices, self.cur_set

    def increase_set_size(self, kappa):
        self.instance_set_size += kappa / len(self.collection_envs)
        return

    def set_instance_set(self, indices):
        size = int(np.ceil(len(self.collection_envs) * self.instance_set_size))
        if size <= 0:
            size = 1
        self.cur_set = np.array(self.collection_envs)[indices[:size]]
        self.cur_set_ids = [indices[:size]]
        self.indices = indices

    def get_context(self):
        return self.contexts[self.cur_id]


class AbstractCurriculumEvalGymWrapper(gym.Env, ABC):
    """Abstract Evaluation Wrapper. It is similar to AbstractCurriculumGymWrapper and
    tasks are selected sequentially for evaluation"""
    def __init__(self, env_type, path):

        self.collection_envs, self.contexts = [], []
        self.ep = -1  # To log episodes
        self.load_envs(path, env_type)
        self.num_steps = 0
        self.num_of_tasks = len(self.collection_envs)
        self.curr_eval_id = -1
        # Random choice of the first task from collection_envs
        self.curr_task = np.random.choice(self.collection_envs)

    def pick_next_id(self):
        self.curr_eval_id = (self.curr_eval_id + 1) % self.num_of_tasks
        return self.curr_eval_id

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def load_envs(self, path, env_type):
        pass


class AbstractSampler(ABC):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        self.noise = 1e-3 * (context_bounds[1] - context_bounds[0])

    def update(self, context: np.ndarray, ret: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.select(samples) + np.random.uniform(-self.noise, self.noise)

    @abstractmethod
    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class UniformSampler(AbstractSampler):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super(UniformSampler, self).__init__(context_bounds)

    def select(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]