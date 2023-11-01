import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Any
import numpy as np
from collections import deque

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

        self.timesteps_batch = TIMESTEPS_BATCH

        self.buffer = []

        self.gamma = GAMMA

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

    def rollout(self):
        batch_data = {'states': [], 'actions': [], 'rewards': [], 'action_probs': [], 'dones': []}
        state = self.env.reset()
        for _ in range(self.timesteps_batch):
            action, action_prob = self.actor_net(state)
            next_state, reward, done = self.env.step(action)

            batch_data['states'].append(state)
            batch_data['actions'].append(action)
            batch_data['rewards'].append(reward)
            batch_data['action_probs'].append(action_prob)
            batch_data['dones'].append(done)

            state = next_state
            if done:
                break

        return batch_data

    # compute Q value
    def compute_accumulated_rewards(self, batch_rewards):
        batch_accumulated_rewards = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_accumulated_rewards.insert(0, discounted_reward)

        batch_accumulated_rewards = torch.tensor(batch_accumulated_rewards, dtype=torch.float)
        return batch_accumulated_rewards

    def evaluate(self, batch_states):
        # compute V value
        v_value = self.critic_net(batch_states)
        # compute log probability of batch_actions using the most recent actor_net
        return v_value



if __name__ == "__main__":
    env = AutoPark_Env()
    agent = Agent(env)