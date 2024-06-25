import csv
import numpy as np
import wandb
from envs.pointmass.pm_sparse.binary_pointmass import BinaryContextualPointMass
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper
import random


class PointmassWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the PointMass environment.
        Inherits from our abstract class so that curriculums can be used"""
    def __init__(self, cur, env_type, beta, target_type, metrics, beta_plr, rho_plr):
        path = "envs/pointmass/task_datasets/pm_train.csv"
        self.type_env = env_type
        self.metrics = metrics
        self.target_type = target_type
        self.contexts_arr = []
        super(PointmassWrapper, self).__init__(cur, env_type, beta, target_type, path, beta_plr, rho_plr)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space_context  # context obs space for PM
        self.context1_ls = []
        self.context2_ls = []
        self.context3_ls = []
        self.empty_env = BinaryContextualPointMass()

    @staticmethod
    def set_context_bounds():
        LOWER_CONTEXT_BOUNDS = np.array([-4., 0.5, 0.])
        UPPER_CONTEXT_BOUNDS = np.array([4., 8., 4.])
        return LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS

    @staticmethod
    # Generate contexts and save them
    def _generate_contexts(num, target_bool=True):
        if target_bool:
            target_task_percentage = 0.00
            with open("task_datasets/pm_train.csv", "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=",")
                writer.writerow(["ID", "c1", "c2", "c3", "target"])
                for i in range(num):
                    if i < (1 - target_task_percentage) * num:
                        context1 = random.uniform(-4, 4)
                        context2 = random.uniform(0.5, 8)
                        context3 = random.uniform(0, 4)
                        target = "false"
                    elif i < (1 - target_task_percentage / 2) * num:
                        context1 = np.random.normal(loc=3.9, scale=0.05)
                        if context1 > 4:
                            context1 = 4
                        context2 = np.random.normal(loc=0.5, scale=0.05)
                        if context2 < 0.5:
                            context2 = 0.5
                        # context3 = abs(np.random.normal(loc=0, scale=0.5))
                        context3 = 0  # random.uniform(0, 4) #random.uniform(0, 4)
                        target = "false"
                    else:
                        context1 = np.random.normal(loc=-3.9, scale=0.05)  # scale=0.5)
                        if context1 < -4:
                            context1 = -4
                        context2 = np.random.normal(loc=0.5, scale=0.05)  # scale=0.5)
                        if context2 < 0.5:
                            context2 = 0.5
                        # context3 = abs(np.random.normal(loc=0, scale=0.5))
                        context3 = 0  # random.uniform(0, 4)
                        target = "true"
                    writer.writerow([i, context1, context2, context3, target])

    @staticmethod
    def generate_single_mode_gaussian_context():

        context1 = np.random.normal(loc=3.9, scale=0.05)
        context2 = np.random.normal(loc=0.5, scale=0.05)
        context3 = random.uniform(0, 4)

        # clip context1 between -4 and 4
        context1 = np.clip(context1, -4, 4)
        # clip context2 between 0.5 and 8
        context2 = np.clip(context2, 0.5, 8)
        return np.array([context1, context2, context3])

    @staticmethod
    def generate_double_mode_gaussian_context():

        # sample a binary value 50% of the time
        if random.random() < 0.5:
            context1 = np.random.normal(loc=3.9, scale=0.05)
            context2 = np.random.normal(loc=0.5, scale=0.05)
            context3 = random.uniform(0, 4)
        else:
            context1 = np.random.normal(loc=-3.9, scale=0.05)
            context2 = np.random.normal(loc=0.5, scale=0.05)
            context3 = random.uniform(0, 4)
        # clip context1 between -4 and 4
        context1 = np.clip(context1, -4, 4)
        # clip context2 between 0.5 and 8
        context2 = np.clip(context2, 0.5, 8)
        return np.array([context1, context2, context3])

    @staticmethod
    def generate_single_task_context():

        context1 = 0.9
        context2 = 0.5
        context3 = 3.5

        return np.array([context1, context2, context3])

    def sampler_interface(self, type):

        if type == "single-mode-gaussian":
            return self.generate_single_mode_gaussian_context()
        elif type == "double-mode-gaussian":
            return self.generate_double_mode_gaussian_context()
        elif type == "single-task":
            return self.generate_single_task_context()
        else:
            raise NotImplementedError

    def generate_one_target_context(self, type):

        sample = self.sampler_interface(type)
        return np.array(sample)

    @staticmethod
    def get_context_state_obs(context):
        state = np.array([0.0, 0.0, 3.0, 0.0])
        context_state = np.concatenate((state, context), axis=0)
        return context_state

    def load_envs(self, path, env_type):
        self.contexts = self.read_csv_tasks(path)
        # Each context is considered a task. We create the environment as a multi-task collection of single tasks.
        for contx in self.contexts:
            context = np.zeros(3)
            context[0] = contx[1]
            context[1] = contx[2]
            context[2] = contx[3]
            self.contexts_arr.append(context)
            # Create a new single environment for each task
            if env_type == "binary":
                single_env = BinaryContextualPointMass(context=context)
            else:
                # Raise error that only binary environments are supported
                print("Only binary environments are supported")
                raise NotImplementedError

            single_env.reset()
            self.collection_envs.append(single_env)
            # Track target tasks
            if contx[4] == "true":
                self.target_tasks.append(int(contx[0]))

        self.contexts_arr = np.array(self.contexts_arr)

    def sim_kernel(self, task_id):
        if not isinstance(self.contexts_arr, np.ndarray):
            self.contexts_arr = np.array(self.contexts_arr)
        dist = np.linalg.norm(self.contexts_arr - self.contexts_arr[task_id], axis=1)

        return np.exp(- dist)

    # Similarity kernel with given context
    def sim_kernel_with_context(self, context):
        if not isinstance(self.contexts_arr, np.ndarray):
            self.contexts_arr = np.array(self.contexts_arr)
        dist = np.linalg.norm(self.contexts_arr - context, axis=1)
        return np.exp(- dist)

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data

    def step(self, action):
        # Task id is updated with the previous selected task when rollout for the new selected task starts
        self.task_id = self.cur_id
        x = self.curr_task.step(action=action)
        self.num_steps += 1
        return x

    def reset(self, seed=None):

        super(PointmassWrapper, self).select_next_task()
        self.num_steps = 0

        if self.metrics:
            # Record features for visualization
            if self.cur == "iid" or self.cur == "procurl-val" or self.cur =="proxcorl":
                _, context1, context2, context3, _ = self.get_context()
                self.context1_ls.append((float(context1)))
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
                wandb.log({"context1": sum(self.context1_ls)/len(self.context1_ls),
                           "context2": sum(self.context2_ls)/len(self.context2_ls),
                           "context3": sum(self.context3_ls)/len(self.context3_ls)})
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


class PointmassWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the PointMass environment """
    def __init__(self, env_type, target_type):
        if target_type == "double-mode-gaussian":
            path = "envs/pointmass/task_datasets/pm_test_heldout.csv"
        elif target_type == "single-mode-gaussian":
            path = "envs/pointmass/task_datasets/pm_test_single_mode_gaussian.csv"
        elif target_type == "single-task":
            path = "envs/pointmass/task_datasets/pm_test_single_task.csv"
        else:
            raise NotImplementedError(f"Target type {target_type} not implemented")
        super(PointmassWrapperEval, self).__init__(env_type, path)
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

            # Create a new environment for each task
            if env_type == "binary":
                single_env = BinaryContextualPointMass(context=context)
            else:
                # Raise error that only binary environments are supported
                print("Only binary environments are supported")
                raise NotImplementedError

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

