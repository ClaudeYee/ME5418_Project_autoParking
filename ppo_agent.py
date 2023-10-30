import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

from model import ActorNet, CriticNet
from Auto_Parking import AutoPark_Env
from parameter import *

# TODO: Remember to modify the reward function to satisfy our target!
class Agent():
    def __init__(self, env):
        # Get the state from the defined env
        self.env = env
        # TODO: the dimensions here might be changed to adjust the dimension defined in our neural network
        self.state_dim = env.world.shape      # [3, 60, 60]
        self.action_dim = env.actions.shape     # [1, 81]   Note that: might be directly assigned to be 18 rather than call the actions property in env class

        # initialize ActorNet and CriticNet
        self.actor_net = ActorNet()
        self.critic_net = CriticNet()

        self.timesteps_batch = TIMESTEPS_BATCH = 480

        self.buffer = []

        self.writer = SummaryWriter("./exp")

    def learn(self, total_t):
        # Timestep so far
        t_so_far = 0
        while t_so_far < total_t:

        # Timesteps run so far in this batch
        t = 0
        while t < self.timesteps_batch:
            # rewards in this episode
            self.env.run_episode(t)


from typing import List, Tuple, Any
import numpy as np
from collections import deque


class Rollout:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(self, state: Any, action: Any, reward: Any, next_state: Any) -> None:

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:

        if batch_size > len(self.buffer):
            raise ValueError("Batch size cannot be greater than the buffer size.")

        sample_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in sample_indices])
        return states, actions, rewards, next_states

    def __len__(self) -> int:

        return len(self.buffer)





if __name__ == "__main__":
    env = AutoPark_Env()
    agent = Agent(env)