import numpy as np
from gym.spaces import Box
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle


class ProCurlTeacher(AbstractTeacher):
    """
       ProCURL teacher that can be added to the TeachMyAgent benchmark to run experiments.
       """
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, beta):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        self.seed = seed
        self.v_max = env_reward_ub
        self.v_min = env_reward_lb
        self.random_task_generator = Box(np.array(mins), np.array(maxs), dtype=np.float32)
        self.random_task_generator.seed(self.seed)
        # Create a buffer of tasks
        self.task_buffer = []
        self.task_buffer_size = 1000
        for i in range(self.task_buffer_size):
            self.task_buffer.append(self.sample_random_task())

        self.posl = np.zeros(len(self.task_buffer))
        self.post = np.ones(len(self.task_buffer))
        self.beta = beta
        self.step_counter = 0
        self.episode_counter = 0

        # Create a dataframe to store the results
        self.df = pd.DataFrame(columns=['context1', "context2", "reward"])
        self.task_dim = 2
        self.regression_model = RegressionModel(task_dim=self.task_dim)
        self.best_weights = None
        self.curriculum = True
        self.selected_tasks = np.empty((0, 2), float)

    def update_buffer(self):
        self.task_buffer = []
        print("Updating the buffer")
        for i in range(self.task_buffer_size):
            self.task_buffer.append(self.sample_random_task())
        self.posl = self.evaluate_contexts()

    def step_update(self, state, action, reward, next_state, done):

        self.step_counter += 1
        if (self.step_counter / 500000 + 1) % 2 == 0:
            self.train_contexts_regressor(self.df)
            self.df = pd.DataFrame(columns=['context1', "context2", "reward"])

        if self.step_counter % 2000000 == 0:
            self.update_buffer()

        else:
            pass

    def init_model(self):
        print("Init model")
        self.regression_model = RegressionModel(task_dim=self.task_dim)

    def train_contexts_regressor(self, context_reward_data):

        self.init_model()
        # Train the model
        X = context_reward_data.drop('reward', axis=1)
        y = context_reward_data['reward']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train_tensor = torch.tensor(X_train.values).float()
        y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)
        X_test_tensor = torch.tensor(X_test.values).float()
        y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

        optimizer = torch.optim.Adam(self.regression_model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # training parameters
        n_epochs = 100  # number of epochs to run
        batch_size = 16  # size of each batch
        batch_start = torch.arange(0, len(X_train_tensor), batch_size)

        # Hold the best model
        best_mse = np.inf  # init to infinity
        best_weights = None
        history = []

        # training loop
        for epoch in range(n_epochs):
            self.regression_model.train()
            with tqdm(batch_start) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train_tensor[start:start + batch_size]
                    y_batch = y_train_tensor[start:start + batch_size]
                    # forward pass
                    y_pred = self.regression_model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            self.regression_model.eval()
            y_pred = self.regression_model(X_test_tensor)
            mse = loss_fn(y_pred, y_test_tensor)
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                # Save best model
                self.best_weights = copy.deepcopy(self.regression_model.state_dict())

    def episodic_update(self, task, reward, is_success):

        # Save task to a pickle file
        if self.curriculum:
            # Append task to the selected tasks
            self.selected_tasks = np.append(self.selected_tasks, [task], axis=0)

            with open(f'curriculum/procurl_tasks_{self.seed}.pkl', 'wb') as f:
                pickle.dump(self.selected_tasks, f)

        # Add the task, the success and the reward to the dataframe df
        self.df = pd.concat([self.df, pd.DataFrame({'context1': [task[0]], 'context2': [task[1]] ,'reward': reward})], ignore_index=True)
        self.posl = self.evaluate_contexts()

        self.episode_counter += 1

    def sample_random_task(self):
        return self.random_task_generator.sample()

    def sample_task(self):

        task = random.choices(population=self.task_buffer,
                              weights=np.exp(self.beta * (self.posl / self.post) * (self.post - self.posl)), k=1)[0]
        return task

    def evaluate_contexts(self):

        # Load the best model
        if self.best_weights is not None:
            self.regression_model.load_state_dict(self.best_weights)
            # List to tensor
            contexts_tensor = torch.tensor(np.array(self.task_buffer)).float()
            self.regression_model.eval()
            posl = self.regression_model(contexts_tensor)
            # Convert to numpy
            posl = posl.detach().numpy().squeeze()
            posl = (posl - self.v_min) / (self.v_max - self.v_min)
            return np.clip(posl, 0, 1)
        else:
            return np.zeros(len(self.task_buffer))

    def non_exploratory_task_sampling(self):
        return {"task": self.sample_task(),
                "infos": {
                    "bk_index": -1,
                    "task_infos": None}
                }

# The regression model used to predict the performance value of the tasks
class RegressionModel(nn.Module):
    def __init__(self, task_dim):
        super(RegressionModel, self).__init__()
        self.layers = nn.Sequential(
                        nn.Linear(task_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
)

    def forward(self, x):
        x = self.layers(x)
        return x