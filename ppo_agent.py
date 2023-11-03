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

        # ----------------------------------------- rollout data shapes ------------------------------------------------ #
        # batch_states: [number of timesteps each batch, states dimensions]
        # batch_actions: [numbwe of timesteps each batch, action dimensions]
        # batch_log_prob(log probability): [number of timesteps each batch]
        # batch_rewards: [number of episodes, number of timesteps per episode]
        # batch_accumulated_rewards: [number of timesteps per batch]
        # batch_episode_length: [number of episodes]
        # -------------------------------------------------------------------------------------------------------------- #
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_valid_actions = []

        self.batch_rewards = []
        self.batch_accumulated_rewards = []         # accumulated_reward refers to reward-to-go in the blog
        self.episode_lengths = []

        self.timesteps_batch = TIMESTEPS_BATCH

        self.buffer = []

        self.gamma = GAMMA

        self.writer = SummaryWriter("./exp")

    def learn(self, total_k):
        # Training epoch
        k = 0
        while k < total_k:
            # Timesteps run so far in this batch, it increments as the timestep increases in one episode, and still increaments in the next episode
            t = 0
            while t < self.timesteps_batch:
                # The following process is done in one batch
                # rewards in this episode
                # TODO: Keep in mind that t cannot get changed through the env.function, hence, must modify
                self.env.run_episode(t)

                self.batch_states.append(env.robot_states)
                self.batch_actions.append(env.actions)
                self.batch_log_probs.append(env.log_probs)
                self.batch_valid_actions.append(env.valid_actions)
                self.batch_rewards.append(env.rewards)
                self.episode_lengths.append(env.episode_length)


                print(self.episode_lengths)

                # Calculate advantage at k-th iteration

                # detach is used to create a independent copy of a tensor
                # batch_rtgs is the reward to get


            batch_states = torch.tensor(self.batch_states, dtype=torch.float)
            batch_actions = torch.tensor(self.batch_actions, dtype=torch.float)
            batch_log_probs = torch.tensor(self.batch_log_probs, dtype=torch.float)
            batch_accumulated_rewards = self.compute_accumulated_rewards(self.batch_rewards)
            batch_valid_actions = torch.tensor(self.batch_valid_actions, dtype=torch.floaty)



            # # Normalizing advantages
            # # 1e-10 is added to prevent a zero Denominator

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                # v_value = self.evaluate(batch_accumulated_rewards)

                v_value, curr_log_probs= self.evaluate(self.batch_states)
                a_value = batch_accumulated_rewards - v_value.detach()
                a_value = (a_value - a_value.mean()) / (a_value.std() + 1e-10)
                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                # subtract the logs is the same as
                # dividing the values and then canceling the log with e^log.
                # TL;DR makes gradient ascent easier behind the scenes.
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                # a_value is the advantage at k-th iteration
                surr1 = ratios * a_value
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * a_value

                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(v_value, batch_accumulated_rewards)

                # Calculate gradients and perform backward propagation for actor network
                # self.actor_optim.zero_grad()
                # actor_loss.backward(retain_graph=True)
                # self.actor_optim.step()
                #
                # # Calculate gradients and perform backward propagation for critic network
                # self.critic_optim.zero_grad()
                # critic_loss.backward()
                # self.critic_optim.step()


    # def rollout(self):
    #     batch_data = {'states': [], 'actions': [], 'rewards': [], 'action_probs': [], 'dones': []}
    #     state = self.env.reset()
    #     for _ in range(self.timesteps_batch):
    #         action, action_prob = self.actor_net(state)
    #         next_state, reward, done = self.env.step(action)
    #
    #         batch_data['states'].append(state)
    #         batch_data['actions'].append(action)
    #         batch_data['rewards'].append(reward)
    #         batch_data['action_probs'].append(action_prob)
    #         batch_data['dones'].append(done)
    #
    #         state = next_state
    #         if done:
    #             break
    #
    #     return batch_data

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
        action_distribution = self.actor_net(batch_states)
        curr_log_probs =
        # compute log probability of batch_actions using the most recent actor_net
        return v_value, curr_log_probs



if __name__ == "__main__":
    env = AutoPark_Env()
    agent = Agent(env)