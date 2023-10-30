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



if __name__ == "__main__":
    env = AutoPark_Env()
    agent = Agent(env)