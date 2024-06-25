import csv
import numpy as np
import wandb
from abstract_classes.AbstractGymWrapper import AbstractCurriculumGymWrapper, AbstractCurriculumEvalGymWrapper
from envs.minigrid.wrappers import FlatObsWrapper
from envs.minigrid.environments import (BlockedUnlockPickupEnv,
                                        UnlockPickupEnv, UnlockEnv,
                                        CrossingEnv, DynamicObstaclesEnv, FourRoomsEnv)


class MinigridWrapper(AbstractCurriculumGymWrapper):
    """ Concrete gym class of the Maze environment.
        Inherits from our abstract class so that curriculums can be used"""
    def __init__(self, cur, beta, target_type, metrics, beta_plr, rho_plr):
        path = None
        self.type_env = "binary"
        self.metrics = metrics
        self.contexts_arr = []
        self.target_type = target_type
        super(MinigridWrapper, self).__init__(cur, self.type_env, beta, target_type, path, beta_plr, rho_plr)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space
        self.context1_ls = []
        self.context2_ls = []
        self.context3_ls = []
        self.context4_ls = []
        self.context5_ls = []
        self.context6_ls = []
        self.context7_ls = []
        self.context8_ls = []
        self.seed = None
        self.single_env = None
        self.empty_env = BlockedUnlockPickupEnv()

    def set_context_bounds(self):
        LOWER_CONTEXT_BOUNDS = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        UPPER_CONTEXT_BOUNDS = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        return LOWER_CONTEXT_BOUNDS, UPPER_CONTEXT_BOUNDS

    def wrap_env(self, env, context):
        return FlatObsWrapper(env, context)

    @staticmethod
    def sampler_interface(type):
        if type == "single-target":
            features = tuple([True, False, False, False, True, True, True, True])
            return features
        else:
            raise ValueError("Invalid target type")

    def generate_one_target_context(self, type):
        sample = self.sampler_interface(type)
        return sample

    @staticmethod
    def map_context(context):

        # Map the context to the closest task
        available_envs = [np.array((1, 1, 1, 0, 0, 0, 0, 0)), np.array((1, 1, 0, 1, 0, 0, 0, 0)),
                          np.array((1, 0, 0, 0, 1, 1, 0, 0)), np.array((1, 0, 0, 0, 0, 0, 0, 0)),
                          np.array((1, 0, 0, 0, 1, 1, 1, 1)), np.array((1, 0, 0, 0, 1, 1, 1, 0)), ]

        env = np.argmin(np.linalg.norm(np.array(available_envs) - context, axis=1))

        if env == 0:
            single_env = CrossingEnv()
        elif env == 1:
            single_env = DynamicObstaclesEnv()
        elif env == 2:
            single_env = UnlockEnv()
        elif env == 3:
            single_env = FourRoomsEnv()
        elif env == 4:
            single_env = BlockedUnlockPickupEnv()
        elif env == 5:
            single_env = UnlockPickupEnv()
        else:
            raise ValueError("Invalid context")

        return single_env

    @staticmethod
    def get_context_state_obs(context):

        if (context == np.array((1, 1, 1, 0, 0, 0, 0, 0))).all():
            env = CrossingEnv()
        elif (context == np.array((1, 1, 0, 1, 0, 0, 0, 0))).all():
            env = DynamicObstaclesEnv()
        elif (context == np.array((1, 0, 0, 0, 1, 1, 0, 0))).all():
            env = UnlockEnv()
        elif (context == np.array((1, 0, 0, 0, 0, 0, 0, 0))).all():
            env = FourRoomsEnv()
        elif (context == np.array((1, 0, 0, 0, 1, 1, 1, 1))).all():
            env = BlockedUnlockPickupEnv()
        elif (context == np.array((1, 0, 0, 0, 1, 1, 1, 0))).all():
            env = UnlockPickupEnv()
        else:
            raise ValueError("Invalid context")

        env = FlatObsWrapper(env, context=context)
        state, _ = env.reset()
        return state

    def load_envs(self, path, env_type):

        num_tasks = 1000
        self.contexts_arr = []
        if path is None:

            for i in range(num_tasks):
                if i < 1.5 * num_tasks / 6:
                    features = np.array((1, 1, 1, 0, 0, 0, 0, 0))
                    single_env = CrossingEnv()
                    print(i, "LavaCrossing")
                elif i < 3.0 * num_tasks / 6:
                    features = np.array((1, 1, 0, 1, 0, 0, 0, 0))
                    single_env = DynamicObstaclesEnv()
                    print(i, "DynamicObstacles")
                elif i < 4.5 * num_tasks / 6:
                    features = np.array((1, 0, 0, 0, 0, 0, 0, 0))
                    single_env = FourRoomsEnv()
                    print(i, "FourRooms")
                elif i < 5.0 * num_tasks / 6:
                    features = np.array((1, 0, 0, 0, 1, 1, 0, 0))
                    single_env = UnlockEnv()
                    print(i, "Unlock")
                elif i < 5.5 * num_tasks / 6:
                    features = np.array((1, 0, 0, 0, 1, 1, 1, 0))
                    single_env = UnlockPickupEnv()
                    print(i, "UnlockPickup")
                else:
                    features = np.array((1, 0, 0, 0, 1, 1, 1, 1))
                    single_env = BlockedUnlockPickupEnv()
                    print(i, "BlockedUnlockPickup")

                self.contexts.append(features)
                wrapped_env = FlatObsWrapper(single_env, context=features)
                wrapped_env.reset(seed=int(i))
                self.collection_envs.append(wrapped_env)
                self.contexts_arr.append(features)
            self.contexts_arr = np.array(self.contexts_arr)

    def sim_kernel_with_context(self, context):
        if not isinstance(self.contexts_arr, np.ndarray):
            self.contexts_arr = np.array(self.contexts_arr).astype(int)
        return np.exp(- np.linalg.norm(self.contexts_arr - np.array(context).astype(int), axis=1))

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
        super(MinigridWrapper, self).select_next_task()
        self.num_steps = 0
        self.seed = seed

        if self.metrics:
            if self.cur == "iid" or self.cur == "procurl-val" or self.cur == "proxcorl":
                context1, context2, context3, context4, context5, context6, context7, context8 = self.get_context()
                self.context1_ls.append(float(context1))
                self.context2_ls.append(float(context2))
                self.context3_ls.append(float(context3))
                self.context4_ls.append(float(context4))
                self.context5_ls.append(float(context5))
                self.context6_ls.append(float(context6))
                self.context7_ls.append(float(context7))
                self.context8_ls.append(float(context8))
            elif self.cur == "currot":
                self.context1_ls.append(float(self.currot_contx[0]))
                self.context2_ls.append(float(self.currot_contx[1]))
                self.context3_ls.append(float(self.currot_contx[2]))
                self.context4_ls.append(float(self.currot_contx[3]))
                self.context5_ls.append(float(self.currot_contx[4]))
                self.context6_ls.append(float(self.currot_contx[5]))
                self.context7_ls.append(float(self.currot_contx[6]))
                self.context8_ls.append(float(self.currot_contx[7]))
            elif self.cur == "target":
                self.context1_ls.append(float(self.target_contx[0]))
                self.context2_ls.append(float(self.target_contx[1]))
                self.context3_ls.append(float(self.target_contx[2]))
                self.context4_ls.append(float(self.target_contx[3]))
                self.context5_ls.append(float(self.target_contx[4]))
                self.context6_ls.append(float(self.target_contx[5]))
                self.context7_ls.append(float(self.target_contx[6]))
                self.context8_ls.append(float(self.target_contx[7]))
            elif self.cur == "gradient":
                self.context1_ls.append(float(self.grad_contx[0]))
                self.context2_ls.append(float(self.grad_contx[1]))
                self.context3_ls.append(float(self.grad_contx[2]))
                self.context4_ls.append(float(self.grad_contx[3]))
                self.context5_ls.append(float(self.grad_contx[4]))
                self.context6_ls.append(float(self.grad_contx[5]))
                self.context7_ls.append(float(self.grad_contx[6]))
                self.context8_ls.append(float(self.grad_contx[7]))
            if len(self.context1_ls) == 1:
                wandb.log({"context1": sum(self.context1_ls) / len(self.context1_ls),
                           "context2": sum(self.context2_ls) / len(self.context2_ls),
                           "context3": sum(self.context3_ls) / len(self.context3_ls),
                           "context4": sum(self.context4_ls) / len(self.context4_ls),
                           "context5": sum(self.context5_ls) / len(self.context5_ls),
                           "context6": sum(self.context6_ls) / len(self.context6_ls),
                           "context7": sum(self.context7_ls) / len(self.context7_ls),
                           "context8": sum(self.context8_ls) / len(self.context8_ls)
                           })
                (self.context1_ls, self.context2_ls, self.context3_ls, self.context4_ls,
                 self.context5_ls, self.context6_ls, self.context7_ls, self.context8_ls) = [], [], [], [], [], [], [], []

        if self.cur == "target" or self.cur == "currot" or self.cur == "gradient":
            return self.curr_task.reset()
        else:
            return self.curr_task.reset(seed=int(self.cur_id))

    def set_env_with_context(self, context, seed):

        if context is None:
            # Here only resetting sets the context
            self.curr_task.reset(seed=int(seed))
        else:
            if self.cur == "target":
                self.curr_task = self.wrap_env(self.empty_env, context)
            else:
                context = np.rint(context)
                single_env = self.map_context(context)
                self.curr_task = self.wrap_env(single_env, context)
                self.curr_task.reset()


class MinigridWrapperEval(AbstractCurriculumEvalGymWrapper):
    """ Concrete evaluation gym class of the Minigrid environment """
    def __init__(self, target_type):
        self.type_env = "binary"
        path = None

        super(MinigridWrapperEval, self).__init__(self.type_env, path)
        self.action_space = self.collection_envs[0].action_space
        self.observation_space = self.collection_envs[0].observation_space
        self.seed = None
        self.single_env = None

    def step(self, action):
        x = self.curr_task.step(action=action)
        return x

    def reset(self, seed=None):
        self.seed = seed
        next_id = self.pick_next_id()
        self.curr_task = self.collection_envs[next_id]
        return self.curr_task.reset()

    def load_envs(self, path, env_type):

        # Target has features of 5 boolean values that are True
        features = np.array((1, 0, 0, 0, 1, 1, 1, 1))

        if path is None:
            for i in range(10):
                self.single_env = BlockedUnlockPickupEnv()
                self.single_env = FlatObsWrapper(self.single_env, context=features)
                self.single_env.reset(seed=int(i+10000))
                self.collection_envs.append(self.single_env)
        else:
            pass

    @staticmethod
    def read_csv_tasks(path):
        with open(path) as f:
            # skip first row
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            data = list(reader)
            return data
