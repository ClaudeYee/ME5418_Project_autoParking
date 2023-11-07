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
    def __init__(self, env, device):
        self.device = device
        # Get the state from the defined env
        self.env = env
        self.env.init_world()
        # TODO: the dimensions here might be changed to adjust the dimension defined in our neural network
        self.state_dim = env.world.shape      # [3, 60, 60]
        self.action_dim = np.array(env.actions).shape     # [1, 81]   Note that: might be directly assigned to be 18 rather than call the actions property in env class

        # initialize ActorNet and CriticNet
        self.actor_net = ActorNet().to(self.device)
        self.critic_net = CriticNet().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=LR)

        # ----------------------------------------- rollout data shapes ---------------------------------------------- #
        # rollout_states: [number of timesteps each rollout, states dimensions]
        # rollout_actions: [numbwe of timesteps each rollout, action dimensions]
        # rollout_log_prob(log probability): [number of timesteps each rollout]
        # rollout_rewards: [number of episodes, number of timesteps per episode]
        # rollout_accumulated_rewards: [number of timesteps per batch]
        # rollout_episode_lengths: [number of episodes]
        # ------------------------------------------ rollout data ---------------------------------------------------- #
        # rollout_states = []   # should be a tensor
        # rollout_actions = []
        # rollout_log_probs = []
        # rollout_valid_actions = []
        # rollout_rewards = []
        # rollout_episode_lengths = []
        self.rollout_data = [[] for _ in range(6)]
        # ----------------------------------------- batch data -------------------------------------------------------#
        self.batch_size = BATCH_SIZE
        self.batch_states = []
        self.batch_actions = []
        self.batch_log_probs = []
        self.batch_valid_actions = []
        self.batch_rewards = []
        self.batch_accumulated_rewards = []         # accumulated_reward refers to reward-to-go in the blog
        self.batch_episode_lengths = []

        self.timesteps_rollout = TIMESTEPS_ROLLOUT
        self.updates_per_iteration = UPDATES_PER_ITERATION      # used to determine how many times pi in numerator

        self.clip = CLIP
        self.gamma = GAMMA

        self.writer = SummaryWriter("./exp")

    def learn(self, total_timesteps):
        k = 0
        while k < total_timesteps:
            # Timesteps run so far in this batch, it increments as the timestep increases in one episode, and still increaments in the next episode
            t = 0
            while t < self.timesteps_rollout:
                # The following process is done in one buffer
                # rewards in this episode
                print("actor_net: ", next(self.actor_net.parameters()).device)
                t = self.env.run_episode(self.actor_net, t)

                # self.batch_states.append(env.robot_states)
                # self.batch_actions.append(env.actions)
                # self.batch_log_probs.append(env.log_probs)
                # self.batch_valid_actions.append(env.valid_actions)
                # self.batch_rewards.append(env.rewards)
                # self.batch_episode_lengths.append(env.episode_length + 1)
                # print(self.batch_episode_lengths)
                # print("env.states: ", env.states.shape)
                self.rollout(env.states, env.actions, env.log_probs, env.valid_actions, env.rewards, env.episode_length + 1)
                # buffer_states.append(env.robot_states)
                # buffer_actions.append(env.actions)
                # buffer_log_probs.append(env.log_probs)
                # buffer_valid_actions.append(env.valid_actions)
                # buffer_rewards.append(env.rewards)
                # buffer_episode_lengths.append(env.episode_length + 1)

                # detach is used to create a independent copy of a tensor

                # How many timesteps it runs in this batch
                k += np.sum(self.rollout_data[5])

            self.env.init_world()

            print("self.env", self.env)

            # rollout prefix but no self are all in tensor form
            print("data has been collected into one rollout")
            # rollout_data = np.array(self.rollout_data)
            # rollout_data = torch.tensor(self.rollout_data, dtype=torch.float).to(self.device)   # [states, actions, log_probs, valid_actions, rewards, episode_lengths]
            # batch_states = torch.tensor(self.batch_states, dtype=torch.float).to(self.device)
            # # batch_actions = torch.tensor(self.batch_actions, dtype=torch.float).to(self.device)
            # batch_rewards = torch.tensor(self.batch_rewards, dtype=torch.float).to(self.device)
            # batch_log_probs = torch.tensor(self.batch_log_probs, dtype=torch.float).to(self.device)
            # # valid_actions may not need to be converted into tensor form. For now, just keep it.
            # batch_valid_actions = torch.tensor(self.batch_valid_actions, dtype=torch.float).to(self.device)

            # compute advantage function value
            self.rollout_data[0] = np.array(self.rollout_data[0]).reshape(-1, 3, WORLD_SIZE[0], WORLD_SIZE[1])
            rollout_states = torch.tensor(self.rollout_data[0], dtype=torch.float).to(self.device)
            print("rollout_states: ", rollout_states.shape)
            rollout_actions = torch.tensor(self.rollout_data[1], dtype=torch.float).to(self.device)
            rollout_log_probs = torch.tensor(self.rollout_data[2], dtype=torch.float).to(self.device)
            rollout_valid_actions = torch.tensor(self.rollout_data[3], dtype=torch.float).to(self.device)
            rollout_rewards = torch.tensor(self.rollout_data[4], dtype=torch.float).to(self.device)
            rollout_accumulated_rewards = self.compute_accumulated_rewards(rollout_rewards).to(self.device)

            # V_phi_k
            v_value, _ = self.evaluate(rollout_states, rollout_valid_actions)

            a_value = rollout_accumulated_rewards.detach() - v_value.detach()
            # advantage normalization
            a_value = (a_value - a_value.mean()) / (a_value.std() + 1e-10)  # 1e-10 is added to prevent zero denominator

            rollout_step = rollout_states.size(0)
            indices = np.array(rollout_step)
            np.random.shuffle(indices)
            # ----- This is the loop where we update our actor_net and critic_net for some n epochs ----- #
            for _ in range(self.updates_per_iteration):  # ALG STEP 6 & 7

                for start in range(0, rollout_step, self.batch_size):

                # Calculate V_phi and pi_theta(a_t | s_t)
                    end = start + self.batch_size
                    index = indices[start:end]
                    batch_states = rollout_states[index]
                    batch_actions = rollout_actions[index]
                    batch_log_probs = rollout_log_probs[index]
                    batch_valid_actions = rollout_valid_actions[index]
                    batch_rewards = rollout_rewards[index]
                    batch_accumulated_rewards = rollout_accumulated_rewards[index]
                    batch_a_value = a_value[index]
                    v_value, curr_log_probs = self.evaluate(batch_states, batch_valid_actions)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # TL;DR makes gradient ascent easier behind the scenes.
                    ratios = torch.exp(curr_log_probs - batch_log_probs)

                    # Calculate surrogate losses.
                    # a_value is the advantage at k-th iteration
                    surr1 = ratios * batch_a_value
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
                    actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), max_norm=0.5, norm_type=2)
                    self.actor_optimizer.step()
                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5, norm_type=2)
                    self.critic_optimizer.step()

            # if
            # batch_rewards.mean().item()



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
        print(batch_states.shape)
        v_value = self.critic_net(batch_states)
        # compute log probability of batch_actions using the most recent actor_net
        action_distribution = self.actor_net(batch_states)
        valid_action_distribution = action_distribution * batch_valid_actions   # product by elements
        normalized_distribution = valid_action_distribution / valid_action_distribution.sum()     # dont know if the sum would be zero

        curr_log_probs = normalized_distribution.log()
        return v_value, curr_log_probs

    # def save(self, checkpoint_path):
    #     print("Model saving...")
    #     # checkpoint = {"policy_model": self.actor_net.state_dict(), "v_value_model": self.critic_net.state_dict(), "policy_optimizer": }
    #     torch.save(checkpoint, checkpoint_path)

    def rollout(self, states, actions, log_probs, valid_actions, rewards, episode_lengths):
        self.rollout_data[0].append(states)
        # print("states: ", self.rollout_data[0].size)
        self.rollout_data[1].append(actions)
        print("actions: ", self.rollout_data[1])
        self.rollout_data[2].append(log_probs)
        self.rollout_data[3].append(valid_actions)
        self.rollout_data[4].append(rewards)
        self.rollout_data[5].append(episode_lengths)


if __name__ == "__main__":
    env = AutoPark_Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device=device)
    agent.learn(1000)
