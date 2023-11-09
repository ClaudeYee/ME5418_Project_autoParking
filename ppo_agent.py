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

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class RolloutBuffer():
    def __init__(self):
        self.states = []
        self.action_indices = []
        self.log_probs = []
        self.valid_actions = []
        self.rewards = []
        self.episode_lengths = []

    def rollout(self, states, action_indices, log_probs, valid_actions, rewards, episode_length, t):
        if t <= TIMESTEPS_ROLLOUT:
            self.add_data(states, action_indices, log_probs, valid_actions, rewards, episode_length)
        else:
            end = TIMESTEPS_ROLLOUT - sum(self.episode_lengths)
            # end_tmp = episode_length - end

            self.add_data(states[:end], action_indices[:end], log_probs[:end],
                          valid_actions[:end], rewards[:end], end)

            tmp_buffer = RolloutBuffer()
            tmp_buffer.add_data(states[end:], action_indices[end:], log_probs[end:],
                          valid_actions[end:], rewards[end:], episode_length - end)
            return tmp_buffer

        return None




    def add_data(self, states, action_indices, log_probs, valid_actions, rewards, episode_length):
        self.states.append(states)
        self.action_indices.append(action_indices)
        self.log_probs.append(log_probs)
        self.valid_actions.append(valid_actions)
        self.rewards.append(rewards)
        self.episode_lengths.append(episode_length)


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
        self.buffer = RolloutBuffer()
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
        tmp_buffer = None
        k = 0
        while k < total_timesteps:
            # Timesteps run so far in this batch, it increments as the timestep increases in one episode, and still increaments in the next episode
            t = 0
            episode_num = 0
            if tmp_buffer != None:
                self.buffer = tmp_buffer
                t = self.buffer.episode_lengths[0]

            while t < self.timesteps_rollout:                   # timesteps_rollout = 512, meaning that inside of which, if episode data exceeds 512, it will be input to another rollout
                # The following process is done in one buffer
                # rewards in this episode
                print("actor_net: ", next(self.actor_net.parameters()).device)
                t = self.env.run_episode(self.actor_net, t)
                print("Episode {} runs {} steps".format(episode_num, env.episode_length))
                episode_num += 1
                # self.batch_states.append(env.robot_states)
                # self.batch_actions.append(env.actions)
                # self.batch_log_probs.append(env.log_probs)
                # self.batch_valid_actions.append(env.valid_actions)
                # self.batch_rewards.append(env.rewards)
                # self.batch_episode_lengths.append(env.episode_length + 1)
                # print(self.batch_episode_lengths)
                # print("env.states: ", env.states.shape)
                tmp_buffer = self.buffer.rollout(env.states, env.actions, env.log_probs, env.valid_actions, env.rewards, env.episode_length, t)
                self.env.init_world()
                # buffer_states.append(env.robot_states)
                # buffer_actions.append(env.actions)
                # buffer_log_probs.append(env.log_probs)
                # buffer_valid_actions.append(env.valid_actions)
                # buffer_rewards.append(env.rewards)
                # buffer_episode_lengths.append(env.episode_length + 1)

                # detach is used to create a independent copy of a tensor

                # How many timesteps it runs in this batch
            k += np.sum(self.buffer.episode_lengths)


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
            buffer_states = np.array(self.buffer.states)
            buffer_states.view([-1, 3, WORLD_SIZE[0], WORLD_SIZE[1]])
            buffer_actions = np.array(self.buffer.actions).reshape(-1, 2)
            buffer_log_probs = np.array(self.buffer.log_probs).reshape(-1, 81)
            buffer_valid_actions = np.array(self.buffer.valid_actions).reshape(-1, 81)
            buffer_rewards = self.buffer.rewards

            buffer_states = torch.tensor(buffer_states, dtype=torch.float)
            # print("rollout_states: ", rollout_states.shape)
            buffer_actions = torch.tensor(buffer_actions, dtype=torch.float)
            buffer_log_probs = torch.tensor(buffer_log_probs, dtype=torch.float)
            buffer_valid_actions = torch.tensor(buffer_valid_actions, dtype=torch.float)
            # rollout_rewards = torch.tensor(rollout_rewards, dtype=torch.float).to(self.device)

            buffer_accumulated_rewards = self.compute_accumulated_rewards(buffer_rewards)

            # V_phi_k
            # old_v_value, _ = self.evaluate_no_grad(rollout_states, rollout_valid_actions)
            #
            # a_value = rollout_accumulated_rewards.detach() - old_v_value.detach()
            # # advantage normalization
            # a_value = (a_value - a_value.mean()) / (a_value.std() + 1e-10)  # 1e-10 is added to prevent zero denominator

            buffer_step = buffer_states.size(0)
            indices = np.array(range(buffer_step))
            np.random.shuffle(indices)
            # ----- This is the loop where we update our actor_net and critic_net for some n epochs ----- #

            for start in range(0, buffer_step, self.batch_size):

            # Calculate V_phi and pi_theta(a_t | s_t)
                end = start + self.batch_size
                index = indices[start:end]
                batch_states = buffer_states[index].to(device)
                batch_actions = buffer_actions[index].to(device)
                batch_log_probs = buffer_log_probs[index].to(device)
                batch_valid_actions = buffer_valid_actions[index].to(device)
                # batch_rewards = rollout_rewards[index]
                batch_accumulated_rewards = buffer_accumulated_rewards[index].to(device)

                batch_old_v_value, _ = self.evaluate(batch_states, batch_valid_actions)
                batch_old_v_value = batch_old_v_value.squeeze(-1)

                batch_a_value = batch_accumulated_rewards.detach() - batch_old_v_value.detach()
                # advantage normalization
                batch_a_value = (batch_a_value - batch_a_value.mean()) / (batch_a_value.std() + 1e-10)  # 1e-10 is added to prevent zero denominator

                for _ in range(self.updates_per_iteration):  # ALG STEP 6 & 7
                    v_value, curr_log_probs = self.evaluate(batch_states, batch_valid_actions)

                    # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                    # TL;DR makes gradient ascent easier behind the scenes.

                    ratios = torch.exp(curr_log_probs - batch_log_probs)

                    # Calculate surrogate losses.
                    # a_value is the advantage at k-th iteration
                    surr1 = torch.mul(ratios, batch_a_value.unsqueeze(-1))                  # ratio: [128, 81]; batch_a_value: [128, 1]
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_a_value.unsqueeze(-1)
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
        # print("batch_rewards: ", batch_rewards.shape)
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
        v_value, _ = self.critic_net(batch_states, batch_states.shape[0])
        v_value = v_value.squeeze(-1)

        # compute log probability of batch_actions using the most recent actor_net
        action_distribution, _ = self.actor_net(batch_states, batch_states.shape[0])
        print("action_dis: ", action_distribution.shape)
        print("valid_actions: ", batch_valid_actions.shape)
        valid_action_distribution = action_distribution * batch_valid_actions   # product by elements
        normalized_distribution = valid_action_distribution / valid_action_distribution.sum()     # dont know if the sum would be zero

        curr_log_probs = normalized_distribution

        return v_value, curr_log_probs

    # def save(self, checkpoint_path):
    #     print("Model saving...")
    #     # checkpoint = {"policy_model": self.actor_net.state_dict(), "v_value_model": self.critic_net.state_dict(), "policy_optimizer": }
    #     torch.save(checkpoint, checkpoint_path)

    # def rollout(self, states, actions, log_probs, valid_actions, rewards, episode_length, t):
    #     if t < self.timesteps_rollout:
    #     # if self.timesteps_rollout < sum(self.rollout_data[5])+episode_length:
    #         self.rollout_data[0].append(states)
    #         # print("states: ", self.rollout_data[0].size)
    #         self.rollout_data[1].append(actions)
    #         print("actions: ", self.rollout_data[1])
    #         self.rollout_data[2].append(log_probs)
    #         self.rollout_data[3].append(valid_actions)
    #         self.rollout_data[4].append(rewards)
    #         self.rollout_data[5].append(episode_length)
    #     else:
    #
    #
    #         # rollout_data = np.array(self.rollout_data)
    #         # return rollout_data
    #     # else:

def show_gpu_memory_taken(one_tensor):
    memory_used = one_tensor.element_size() * one_tensor.nelement()
    memory_used = memory_used / (1024 * 1024)
    print("It takes {} Mb.".format(memory_used))

if __name__ == "__main__":
    env = AutoPark_Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    agent = Agent(env, device=device)
    agent.learn(1000)
