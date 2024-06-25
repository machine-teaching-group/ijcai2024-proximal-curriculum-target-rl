import csv
import numpy as np
import os
import wandb
import random
from envs.sgr.maze import MazeEnv
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper


class SgrWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the Maze environment.
        Inherits from our abstract class so that curriculums can be used"""
    def __init__(self, cur, beta, target_type, metrics, beta_plr, rho_plr):
        path = "envs/sgr/task_datasets/sgr_train.csv"
        self.type_env = "binary"
        self.metrics = metrics
        self.contexts_arr = []
        self.target_type = target_type
        super(SgrWrapper, self).__init__(cur, self.type_env, beta, target_type, path, beta_plr, rho_plr)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space_context
        self.context1_ls = []
        self.context2_ls = []
        self.context3_ls = []
        self.empty_env = MazeEnv()

    @staticmethod
    def set_context_bounds():
        LOWER_CONTEXT_BOUNDS = np.array([-9., -9., 0.05])
        UPPER_CONTEXT_BOUNDS = np.array([9., 9., 18.])
        return  LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS

    @staticmethod
    # Generate contexts and save them
    def _generate_contexts(num, target_bool=True):
        if target_bool:
            target_task_percentage = 0.01
            with open("task_datasets/sgr_train.csv", "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(["ID", "pos X", "pos Y", "threshold", "target"])
                for i in range(num):
                    if i < (1 - target_task_percentage) * num:
                        lower_context_bounds = np.array([-9., -9., 0.05])
                        upper_context_bounds = np.array([9., 9., 18])
                        sample = np.random.uniform(lower_context_bounds, upper_context_bounds)
                        target = "false"
                    else:
                        lower_context_bounds = np.array([-9., -9., 0.05])
                        upper_context_bounds = np.array([9., 9., 0.05])
                        sample = np.random.uniform(lower_context_bounds, upper_context_bounds)
                        while not MazeEnv._is_feasible(sample):
                            sample = np.random.uniform(lower_context_bounds, upper_context_bounds)
                        target = "true"
                    writer.writerow([i, sample[0], sample[1], sample[2], target])

    @staticmethod
    def generate_single_mode_gaussian_context():
        context1 = random.uniform(-9, 9)
        context2 = random.uniform(-9, 9)
        context3 = np.random.normal(0.06, 0.01)
        # Clip context3 to be above 0.05
        if context3 < 0.05:
            context3 = 0.05
        return np.array([context1, context2, context3])

    @staticmethod
    def generate_double_mode_gaussian_context():
        context1 = random.uniform(-9, 9)
        context2 = random.uniform(-9, 9)
        if random.random() < 0.5:
            context3 = np.random.normal(0.06, 0.01)
        else:
            context3 = np.random.normal(0.12, 0.01)
        # Clip context3 to be above 0.05
        if context3 < 0.05:
            context3 = 0.05
        return np.array([context1, context2, context3])

    # Original sgr target
    @staticmethod
    def generate_single_plane_context():
        lower_context_bounds = np.array([-9., -9., 0.05])
        upper_context_bounds = np.array([9., 9., 0.05])
        return np.random.uniform(lower_context_bounds, upper_context_bounds)

    @staticmethod
    def generate_single_task_context():
        return np.array([6.0, 6.0, 0.05])

    def sampler_interface(self, type):

        if type == "single-mode-gaussian":
            return self.generate_single_mode_gaussian_context()
        elif type == "double-mode-gaussian":
            return self.generate_double_mode_gaussian_context()
        elif type == "single-plane":
            return self.generate_single_plane_context()
        elif type == "single-task":
            return self.generate_single_task_context()
        else:
            raise ValueError("Invalid type")

    def generate_one_target_context(self, type):

        sample = self.sampler_interface(type)
        return np.array(sample)

    @staticmethod
    def get_context_state_obs(context):
        state = MazeEnv().sample_initial_state()
        context_state = np.concatenate((state, context), axis=0)
        return context_state

    def load_envs(self, path, env_type):

        # If no context exist generate them. Otherwise read them
        if os.path.exists(path):
            self.contexts = self.read_csv_tasks(path)
        else:
            num_of_tasks = 10000
            self._generate_contexts(num_of_tasks)
            self.contexts = self.read_csv_tasks(path)

        # Each context is considered a task. We create the environment as a multi-task collection of single tasks.
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]
            self.contexts_arr.append(context)
            # Create a new environment for each task
            self.single_env = MazeEnv(context=context)
            self.single_env.reset()
            self.collection_envs.append(self.single_env)
            # Track target tasks
            if contx[4] == "true":
                self.target_tasks.append(int(contx[0]))
                self.target_contexts.append(context)

        self.contexts_arr = np.array(self.contexts_arr)

    def sim_kernel(self, task_id):
        if not isinstance(self.contexts_arr, np.ndarray):
            self.contexts_arr = np.array(self.contexts_arr)
        return np.exp(- np.linalg.norm(self.contexts_arr - self.contexts_arr[task_id], axis=1))

    def sim_kernel_with_context(self, context):
        if not isinstance(self.contexts_arr, np.ndarray):
            self.contexts_arr = np.array(self.contexts_arr)
        return np.exp(- np.linalg.norm(self.contexts_arr - context, axis=1))

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

    def step(self, action):
        self.num_steps += 1
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        x = self.curr_task.step(action=action)
        return x

    def reset(self, seed=None):
        super(SgrWrapper, self).select_next_task()
        self.num_steps = 0

        if self.metrics:
            if self.cur == "iid" or self.cur == "procurl-val" or self.cur == "proxcorl":
                _, context1, context2, context3, _ = self.get_context()
                self.context1_ls.append(float(context1))
                self.context2_ls.append(float(context2))
                self.context3_ls.append(float(context3))
            elif self.cur == "currot":
                self.context1_ls.append(float(self.currot_contx[0]))
                self.context2_ls.append(float(self.currot_contx[1]))
                self.context3_ls.append(float(self.currot_contx[2]))
            elif self.cur == "target":
                self.context1_ls.append(float(self.target_contx[0]))
                self.context2_ls.append(float(self.target_contx[1]))
                self.context3_ls.append(float(self.target_contx[2]))
            elif self.cur == "gradient":
                self.context1_ls.append(float(self.grad_contx[0]))
                self.context2_ls.append(float(self.grad_contx[1]))
                self.context3_ls.append(float(self.grad_contx[2]))
            if len(self.context1_ls) == 1:
                wandb.log({"context1": sum(self.context1_ls) / len(self.context1_ls),
                           "context2": sum(self.context2_ls) / len(self.context2_ls),
                           "context3": sum(self.context3_ls) / len(self.context3_ls)})
                self.context1_ls, self.context2_ls, self.context3_ls = [], [], []

        return self.curr_task.reset()

    def set_env_with_context(self, context, seed):

        if context is None:
            # Here only resetting sets the context
            self.curr_task.reset()
        else:
            self.empty_env.set_context(context)
            self.curr_task = self.empty_env
            self.curr_task.reset()


class SgrWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the SGR environment """
    def __init__(self, target_type):
        self.type_env = "binary"
        if target_type == "single-plane":
            path = "envs/sgr/task_datasets/sgr_test_heldout.csv"
        elif target_type == "single-mode-gaussian":
            path = "envs/sgr/task_datasets/sgr_test_single_mode_gaussian.csv"
        elif target_type == "double-mode-gaussian":
            path = "envs/sgr/task_datasets/sgr_test_double_mode_gaussian.csv"
        elif target_type == "single-task":
            path = "envs/sgr/task_datasets/sgr_test_single_task.csv"
        else:
            raise NotImplementedError(f"Target type {target_type} not implemented")
        super(SgrWrapperEval, self).__init__(self.type_env, path)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space_context

    def step(self, action):
        x = self.curr_task.step(action=action)
        return x

    def reset(self, seed=None):
        next_id = self.pick_next_id()
        self.curr_task = self.collection_envs[next_id]
        return self.curr_task.reset()

    def load_envs(self, path, env_type):
        self.contexts = self.read_csv_tasks(path)
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]
            single_env = MazeEnv(context=context)
            single_env.reset()
            self.collection_envs.append(single_env)

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data