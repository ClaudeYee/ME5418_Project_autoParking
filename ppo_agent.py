import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR)

        # ----------------------------------------- rollout data shapes ---------------------------------------------- #
        # batch_states: [number of timesteps each batch, states dimensions]
        # batch_actions: [numbwe of timesteps each batch, action dimensions]
        # batch_log_prob(log probability): [number of timesteps each batch]
        # batch_rewards: [number of episodes, number of timesteps per episode]
        # batch_accumulated_rewards: [number of timesteps per batch]
        # batch_episode_lengths: [number of episodes]
        # ------------------------------------------------------------------------------------------------------------ #
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_valid_actions = []

        self.batch_rewards = []
        self.batch_accumulated_rewards = []         # accumulated_reward refers to reward-to-go in the blog
        self.batch_episode_lengths = []

        self.timesteps_batch = TIMESTEPS_BATCH
        self.updates_per_iteration = UPDATES_PER_ITERATION      # used to determine how many times pi in numerator

        self.clip = CLIP
        self.gamma = GAMMA

        self.writer = SummaryWriter("./exp")

    def learn(self, total_timesteps):
        k = 0
        while k < total_timesteps:
            # Timesteps run so far in this batch, it increments as the timestep increases in one episode, and still increaments in the next episode
            # TODO: how to change the value of t in run_episode ???
            t = 0
            while t < self.timesteps_batch:
                # The following process is done in one batch
                # rewards in this episode
                # TODO: Keep in mind that t cannot get changed through the env.function, hence, must modify
                t = self.env.run_episode(self.actor_net, t)

                self.batch_states.append(env.robot_states)
                self.batch_actions.append(env.actions)
                self.batch_log_probs.append(env.log_probs)
                self.batch_valid_actions.append(env.valid_actions)
                self.batch_rewards.append(env.rewards)
                self.batch_episode_lengths.append(env.episode_length + 1)
                print(self.batch_episode_lengths)
                # detach is used to create a independent copy of a tensor

                # How many timesteps it runs in this batch
                k += np.sum(self.batch_episode_lengths)

            # variables with batch prefix but no self are all in tensor form
            batch_states = torch.tensor(self.batch_states, dtype=torch.float)
            batch_actions = torch.tensor(self.batch_actions, dtype=torch.float)
            batch_rewards = torch.tensor(self.batch_rewards, dtype=torch.float)
            batch_log_probs = torch.tensor(self.batch_log_probs, dtype=torch.float)
            batch_accumulated_rewards = self.compute_accumulated_rewards(self.batch_rewards)
            # valid_actions may not need to be converted into tensor form. For now, just keep it.
            batch_valid_actions = torch.tensor(self.batch_valid_actions, dtype=torch.float)

            # compute advantage function value
            # V_phi_k
            v_value, _ = self.evaluate(batch_states, batch_valid_actions)

            a_value = batch_accumulated_rewards - v_value.detach()
            # advantage normalization
            a_value = (a_value - a_value.mean()) / (a_value.std() + 1e-10)  # 1e-10 is added to prevent zero denominator

            # ----- This is the loop where we update our actor_net and critic_net for some n epochs ----- #
            for _ in range(self.updates_per_iteration):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                v_value, curr_log_probs = self.evaluate(batch_states, batch_valid_actions)

                # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
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
                mseloss = nn.MSELoss()
                critic_loss = mseloss(v_value, batch_accumulated_rewards)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


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

    # compute accumulated rewards
    def compute_accumulated_rewards(self, batch_rewards):
        batch_accumulated_rewards = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_accumulated_rewards.insert(0, discounted_reward)

        batch_accumulated_rewards = torch.tensor(batch_accumulated_rewards, dtype=torch.float)
        return batch_accumulated_rewards

    def evaluate(self, batch_states, batch_valid_actions):
        # batch_states and batch_valid_actions are both in the form of tensor
        # compute V value
        v_value = self.critic_net(batch_states)
        # compute log probability of batch_actions using the most recent actor_net
        action_distribution = self.actor_net(batch_states)
        valid_action_distribution = action_distribution * batch_valid_actions   # product by elements
        normalized_distribution = valid_action_distribution / valid_action_distribution.sum()     # dont know if the sum would be zero

        curr_log_probs = normalized_distribution.log()
        return v_value, curr_log_probs


if __name__ == "__main__":
    env = AutoPark_Env()
    agent = Agent(env)
