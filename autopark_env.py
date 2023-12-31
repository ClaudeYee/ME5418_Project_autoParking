import os.path

import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
import sys
import time
import matplotlib.pyplot as plt
import random
import math
import copy
# from gym.envs.classic_control import rendering
from obs_pklots_generator import generate_obs, generate_pklots
from model import ActorNet

from parameter import *
import torch
from torch.distributions import Categorical

# from test2 import ActorNet, CriticNet

'''
    Observation: (position maps of current agent, current goal, obstacles), vector to goal (vx, vy, norm_v)

    Action space: (Tuple)
        	agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''


# opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
# JOINT = False # True for joint estimation of rewards for closeby agents
# dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
# actionDict={v:k for k,v in dirDict.items()}

class State():
    def __init__(self, world_obs_pklot, pos, carSize=ROBOT_SIZE):
        # assert (len(world0.shape) == 2 and world0.shape == goals.shape)
        self.state = world_obs_pklot.copy()
        self.current_pos = pos.copy()
        self.next_pos = -1 * np.ones(WORLD_SIZE)
        self.robot_current_state = self.getState()  # TODO: This might not be needed later.
        self.robot_next_state = self.robot_current_state.copy()
        self.robot_size = carSize
        self.shape0, self.shape1 = self.getShape(carSize)  # TODO: here do some changes
        self.next_hitbox_index = self.getHitBox_index(self.robot_current_state[0], self.robot_current_state[1])
        self.next_hitbox = self.renderHitBox()
        self.current_hitbox_index = self.next_hitbox_index.copy()
        self.current_hitbox = self.next_hitbox.copy()
        self.num_translation_actions = 9
        # 0: Stay, 1: East, 2: Northeast, 3: North, 4: Northwest, 5: West, 6: Southwest, 7: South, 8: Southeast
        self.num_rotation_actions = 3
        # 0: Stay, 1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315 (degree)

        # self.action_space = spaces.Tuple((spaces.Discrete(self.num_translation_actions), spaces.Discrete(self.num_rotation_actions)))
        # define action space two action:translation and rotation

        self.translation_directions = {
            0: (0, 0),  # STAY
            1: (1, 0),  # EAST
            2: (1, 1),  # NORTHEAST
            3: (0, 1),  # NORTH
            4: (-1, 1),  # NORTHWEST
            5: (-1, 0),  # WEST
            6: (-1, -1),  # SOUTHWEST
            7: (0, -1),  # SOUTH
            8: (1, -1)  # SOUTHEAST
        }

    # # def scanForAgent(self):
    # #     agent_pos = (-1, -1)
    # #     agent_last = (-1, -1)
    # #     for i in range(self.state.shape[0]):
    # #         for j in range(self.state.shape[1]):
    # #             if (self.state[i, j] > 0):
    # #                 agent_pos = (i, j)
    # #                 agent_last = (i, j)
    # #             if (self.goals[i, j] > 0):
    # #                 agent_goal = (i, j)
    #
    #     assert (agent_pos != (-1, -1) and agent_goal != (-1, -1))
    #     assert (agent_pos == agent_last)
    #     return agent_pos, agent_last, agent_goal

    # def setDirection(self, direction):
    #     self.direction = direction

    # Scan the pos matrix and return the coordinate of the robot,
    # The value at this position is 1, and dir refers to the current direction of the robot
    def getState(self):
        size = [np.size(self.current_pos, 0), np.size(self.current_pos, 1)]
        for i in range(size[0]):
            for j in range(size[1]):
                if self.current_pos[i, j] != -1:
                    return [[i, j], self.current_pos[i, j]]

    # Try to move agent and return the status
    def moveValidity(self, action):
        # action is a list. Its first element is destination,
        # And its second element is desired next_dir
        next_pos, next_dir = self.get_new_coord_and_rotation_index_from_action(action)

        # Not moving is always allowed
        if next_pos == self.robot_current_state[0] and next_dir == self.robot_current_state[1]:
            return 0

        # Otherwise, let's look at the validity of the move
        hitbox_index = self.getHitBox_index(next_pos, next_dir)

        for i in range(len(hitbox_index)):
            x, y = hitbox_index[i][0], hitbox_index[i][1]
            if (x > self.state.shape[0] - 1 or x < 0
                    or y > self.state.shape[1] - 1 or y < 0):  # out of bounds
                return -1

            if self.state[x, y] == (-1):  # collide with static obstacle
                return -2

        # none of the above
        return 0

    def moveAgent(self, action):
        next_pos, next_dir = self.get_new_coord_and_rotation_index_from_action(action)

        # refresh robot_current_state, pos and hitbox
        self.robot_current_state = self.robot_next_state.copy()
        self.current_pos = self.next_pos.copy()
        self.current_hitbox_index = self.next_hitbox_index.copy()
        self.current_hitbox = self.next_hitbox.copy()

        # Valid move: we can carry out the action in next_pos & robot_state
        self.next_pos[self.robot_next_state[0]] = -1
        self.robot_next_state[0] = next_pos
        self.robot_next_state[1] = next_dir
        self.next_pos[self.robot_next_state[0]] = next_dir

        # Get the new hit box for next stage
        self.next_hitbox_index = self.getHitBox_index(self.robot_next_state[0], self.robot_next_state[1])
        self.next_hitbox = self.renderHitBox()

    def parking_complete(self):
        # Scan the first pixel in next_hitbox_index
        x, y = self.next_hitbox_index[0][0], self.next_hitbox_index[0][1]
        n = self.state[x, y]
        # If it's not in parking space, then our robot is not parked.
        if n == 0:
            return 0
        # If it is in one of the parking space, check if the rest pixels is in the same.
        pklot_id = n

        # Check the rest pixels
        for i in range(len(self.next_hitbox_index)):
            x, y = self.next_hitbox_index[i][0], self.next_hitbox_index[i][1]
            n = self.state[x, y]
            if n != pklot_id:
                return 0

        return int(pklot_id)

    def sample_action(self):
        # sampling actions
        action_index = [random.randint(0, 8), random.randint(0, 2)]
        if action_index == None:
            action_index = self.sample_action()
        return action_index

    def get_action_data(self, action):
        # return data
        return action

    def get_new_coord_and_rotation_index_from_action(self, action):
        # Here action is a list, the first element is the index for transition,
        # And the second is s the index for rotation.
        # The function generate new_coord for moveValidity.
        a = action[0]
        translation = self.translation_directions[a]
        if action[1] == 0:
            rotation = self.robot_next_state[1]
        elif action[1] == 1:
            rotation = self.robot_next_state[1] - 1
        elif action[1] == 2:
            rotation = self.robot_next_state[1] + 1
        if rotation < 0:
            rotation += 8
        elif rotation >= 8:
            rotation -=8
        new_coord = (self.robot_next_state[0][0] + translation[0], self.robot_next_state[0][1] + translation[1])
        return new_coord, rotation

    # def getDir(self, action):
    #     return dirDict[action]
    #
    # def getAction(self, direction):
    #     return actionDict[direction]

    # Space occupied by the vehicle is represented by hitbox
    # It is a list with every element a 1*2 vector,
    # The vector is the index of a pixel, which the vehicle occupies
    # This function will generate 2 hitboxes, others will be obtained
    # by a simple rotate & translation transformation
    # TODO: I suggest this this function should take in the carSize and its direction to determine the output, which is whether shape0 or shape1
    def getShape(self, carSize):
        # shape0 is the hitbox when car is at [0, 0] with dir = 0
        # shape1 is the hitbox with dir = 1
        carShape = [int((carSize[1] - 1) / 2), int((carSize[1] - 1) / 2), int((carSize[0] - 1) / 2)]
        shape0 = []
        shape1 = []

        # carShape is a three element tuple
        # The first one element is the distant from the centre to the front edge of car
        # The second is the distant to rear edge, the third is to left/right edge
        for i in range(-1 * carShape[2], carShape[2] + 1, 1):
            for j in range(-1 * carShape[1], carShape[0] + 1, 1):
                shape0.append([i, j])

        # calculate the max possible size after rotation
        maxL = np.power(carShape[1] + carShape[0] + 1, 2) + np.power(2 * carShape[2] + 1, 2)
        maxL = int(np.ceil(np.power(maxL, 0.5))) + 3

        angle_rad = np.radians(45)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        for y in range(-maxL, maxL + 1, 1):
            for x in range(-maxL, maxL + 1, 1):
                pixel_coords = np.array([x, y])
                # rotate pixel back to origin coords
                rotated_pixel_coords = pixel_coords.dot(np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]]))

                # see if within origin hitbox
                if ((-1 * carShape[1]) - 0.5 < rotated_pixel_coords[0] < (carShape[0] + 0.5) and
                        (-1 * carShape[2]) - 0.5 < rotated_pixel_coords[1] < (carShape[2]) + 0.5):
                    # if so, add to shape1
                    shape1.append([x, y])
        return shape0, shape1

    # Calculate the hitbox
    def getHitBox_index(self, pos, dir):
        # desired position and direction
        agent_pos = pos
        agent_dir = dir

        shift = np.array(agent_pos)
        hitbox_index = []

        # See if direction is tilted
        # If not, use Shape0
        if agent_dir % 2 == 0:
            shape = self.shape0
            angle = agent_dir / 2 * 90
        # Else, use Shape1
        else:
            shape = self.shape1
            angle = (agent_dir - 1) / 2 * 90

        # Calculate the rotation Matrix
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotateMatrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        for i in range(len(shape)):
            # Apply the rotation Transform
            rotated_pixels = np.array(shape[i]).dot(rotateMatrix)
            # Apply the translation Transform
            finial_pixels = rotated_pixels + shift
            hitbox_index.append([round(finial_pixels[0]), round(finial_pixels[1])])

        return hitbox_index

    def renderHitBox(self):
        hitbox = np.zeros([self.state.shape[0], self.state.shape[1]])
        for i in range(len(self.next_hitbox_index)):
            index0 = self.next_hitbox_index[i][0]
            index1 = self.next_hitbox_index[i][1]
            hitbox[index0, index1] = 1
        return hitbox


class AutoPark_Env(gym.Env):
    def __init__(self, world0=None, blank_world=False):  # blank_world: there is no robot and any parking lots
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.world_size = WORLD_SIZE
        self.robot_size = ROBOT_SIZE
        self.parklot_size = PARKLOT_SIZE

        # tunable parameters for dist_reward
        self.dist_reward_param = DIST_REWARD_PARAM

        self.world_obs = None
        self.world_pklot = None
        self.world_robot = None
        self.world = None  # world has two channels: world_obs, world_pklot, world_robot. (3, 60, 60)

        self.init_robot_pos = None
        self.init_robot_dir = None

        self.robot_pos = None
        self.robot_dir = None

        self.pklot1_coord = None  # shape: [1, 2]
        self.pklot2_coord = None  # shape: [1, 2]

        # self.init_world(world0)
        self.init_robot_state = None  # might have some problems here

        # save the robot state, the action, and the reward at each timestep for this episode
        self.robot_states = []
        # self.states = torch.tensor([], dtype=torch.float).to(self.device)
        self.states = []
        self.action_indexs =[]
        self.actions = []
        self.log_probs = []
        self.valid_actions = []
        self.rewards = []  # Note: rewards here refers to the rewards in one episode, shape: [this episode length]
        self.accumulated_rewards = []
        # NOTE: episode_length could be less (task completed) or equal to max_episode_length (task failed)
        self.episode_length = 0
        self.max_episode_length = MAX_EPISODE_LENGTH

        self.done = False

        self.img_save_path = None

    # randomly generate a world with obstacles and two parking lots. Note that they are separately put into two channels of the same world
    def init_world(self):
        # if world0:
        #     world = world0.copy()
        # else:
        self.world_obs = self.init_obstacles()  # the obstacle channel of the world
        # self.world_obs = np.zeros(WORLD_SIZE)
        self.world_pklot, self.pklot1_coord, self.pklot2_coord = self.init_pklots()
        self.robot_pos, self.robot_dir, self.world_robot = self.init_robot(self.world_obs,
                                                                           self.world_pklot)  # NOTE: self.world_robot is robot_hitbox
        self.world = np.array([self.world_obs, self.world_pklot, self.world_robot])

    def init_obstacles(self):
        world_obs = generate_obs(size=self.world_size)
        return world_obs

    def init_pklots(self):
        world_pklots, self.pklot1_coord, self.pklot2_coord = generate_pklots(self.world_size, self.parklot_size,
                                                                             self.world_obs)
        return world_pklots, self.pklot1_coord, self.pklot2_coord

    # Place the robot into the environment in a random position if the position and any grid that the robot takes are not in the girds of obstacles
    def init_robot(self, world_obs, world_pklot):
        self.episode_length = 0

        # pos_x and pos_y refer to the center point coordinate of the robot
        coord_x, coord_y = np.random.randint(3, world_obs.shape[0] - 3), np.random.randint(3, world_obs.shape[1] - 3)
        # randomly generate a heading of the robot
        dir = random.randint(0, 7)
        init_robot_pos_coord = [coord_x, coord_y]
        init_robot_dir = dir
        init_robot_pos = -1 * np.ones([world_obs.shape[0], world_obs.shape[1]])
        init_robot_pos[init_robot_pos_coord[0], init_robot_pos_coord[1]] = init_robot_dir

        init_robot_state = State(world_pklot - world_obs, init_robot_pos)
        init_robot_hitbox = init_robot_state.next_hitbox

        # determine whether the robot center has been placed in the free space
        # TODO: this is defined in robot channel of the whole map (there are other channels)
        if (world_obs[coord_x, coord_y] == 0) and check_available(init_robot_hitbox, world_obs):
            # if check_available(init_robot_hitbox, world_obs):
            # init_robot_pos_coord = [pos_x, pos_y]
            # init_robot_pos = -1 * np.ones([world.shape[0], world.shape[1]])
            # init_robot_pos[init_robot_pos_coord[0], init_robot_pos_coord[1]] = init_robot_dir
            # init_robot_dir = dir
            # init_robot_hitbox = init_hitbox
            self.init_robot_state = init_robot_state
            return init_robot_pos, init_robot_dir, init_robot_hitbox
        else:
            # print("not available")
            init_robot_pos, init_robot_dir, init_robot_hitbox = self.init_robot(world_obs, world_pklot)
            return init_robot_pos, init_robot_dir, init_robot_hitbox

    def step(self, actor_net, robot_state, done=False):
        # If the time step is still not done we can verify if the action is valid and if yes we can complete the action
        # and change the state of our robot and the different parameters accordingly
        ## ------- For now, randomly sample an action from the valid action space for testing without training ------- ##
        # action = select_valid_action(robot_state)
        state, action, action_index, log_prob, valid_action = self.act(actor_net, robot_state)
        self.action_indexs.append(action_index)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        # robot_state transition
        old_robot_coord = robot_state.getState()
        if not old_robot_coord:
            old_robot_coord = [0,0]
        else:
            old_robot_coord = old_robot_coord[0]
        robot_state.moveAgent(action_index)
        next_robot_coord, next_robot_dir = robot_state.get_new_coord_and_rotation_index_from_action(action_index)

        if robot_state.parking_complete() == 1 or robot_state.parking_complete() == 2:
            done = True
        # robot receives reward after conducting an action
        reward = self.compute_reward(old_robot_coord, next_robot_coord, action_index, done)

        # robot_state is a class, while state is a 3x60x60 matrix
        return robot_state, state, reward, done, valid_action

    def run_episode(self, actor_net, t,
                    episode_index):  # add t to record the number of timesteps run so far in this batch
        done = False
        self.reset()

        robot_state = copy.deepcopy(self.init_robot_state)  # start the first step from the init_robot_state, and set it as the robot_current_state
        self.robot_states.append(robot_state)  # save the initial robot state

        for i in range(self.max_episode_length):
            # increment timesteps run this batch so far
            t += 1
            self.episode_length += 1
            # if the task is not completed or not reach episode_length, do step for state transition and save robot_state and reward
            robot_state, state, reward, done, valid_action = self.step(actor_net=actor_net, robot_state=robot_state,
                                                                       done=done)
            self.world_robot = robot_state.next_hitbox
            self.world = [self.world_obs, self.world_pklot, self.world_robot]

            state = state.squeeze(0)
            self.states.append(state)

            self.valid_actions.append(valid_action)  # save the vector of valid actions. 1 for valid.
            self.robot_states.append(robot_state)  # save the state for each move
            self.rewards.append(reward)  # save the reward for each move

            self.plot_env(step=i, episode_index=episode_index)

            if done:
                # self.save_accumulated_reward(self.accumulated_reward)
                print("The robot has successfully parked in the parking lot, task succeeded!")
                return t

        if not done:
            print("The steps in this episode have exceeded the episode length we set, task failed.")
        # print("self.states", self.states.shape)

        return t

    # TODO: Let's define it later, because for now, the design of reward mechanism is not so important for env testing.
    # TODO: NOTE: This part is extremely important for training the agent!!! It might be revised for many times.
    def compute_reward(self, old_robot_coord, new_robot_coord, action_index, done):
        # 1. Reward for task completion
        if done:
            final_reward = FINAL_REWARD  # if the robot successfully has parked into the the parking lot, it receives a very big reward
            reward = final_reward
        else:
            translate_cost = 0
            rotate_cost = 0
            dist_reward = 0
            if action_index[1] == 0 and action_index[0] == 0:
                return -10
            if action_index[1] != 0:
                rotate_cost = ROTATE_COST
            if action_index[0] != 0:
                translate_cost = TRANSLATE_COST

                # 2. Reward to stimulate the robot to get closer to the parking lots (either parking lot 1 or parking
                # lot 2) a = np.array(list(robot_pos)) - np.array(self.pklot1_coord)
                dist_to_pklot1 = np.linalg.norm(np.array(list(new_robot_coord)) - np.array(self.pklot1_coord), ord=1)
                dist_to_pklot2 = np.linalg.norm(np.array(list(new_robot_coord)) - np.array(self.pklot2_coord), ord=1)
                old_dist_to_pklot1 = np.linalg.norm(np.array(old_robot_coord) - np.array(self.pklot1_coord), ord=1)
                old_dist_to_pklot2 = np.linalg.norm(np.array(old_robot_coord) - np.array(self.pklot2_coord), ord=1)
                # new_dist_to_pklot1 = np.linalg.norm(np.array)

                # calculate the angle between robot heading and the link to the parkinglot center
                # move_angle = (action_index[0] - 1) * np.pi / 4
                # move_vector = np.array([np.cos(move_angle), np.sin(move_angle)])
                # vector_to_pklot1 = np.array(self.pklot1_coord) - np.array(list(robot_coord))
                # vector_to_pklot2 = np.array(self.pklot2_coord) - np.array(list(robot_coord))
                #
                # # calculate the sin value of the angles
                # cos_to_pklot1 = np.dot(vector_to_pklot1, move_vector) / (
                #             np.linalg.norm(move_vector) * np.linalg.norm(vector_to_pklot1) + 1e-10)
                # cos_to_pklot2 = np.dot(vector_to_pklot2, move_vector) / (
                #             np.linalg.norm(move_vector) * np.linalg.norm(vector_to_pklot2) + 1e-10)
                #
                # # dist_reward_param is a tunable parameter to guarantee tht dist_reward will never greater than the final reward
                # dist_reward1 = (cos_to_pklot1) * np.sqrt(self.dist_reward_param / (dist_to_pklot1 + 20))
                #
                # dist_reward2 = (cos_to_pklot2) * self.dist_reward_param / (dist_to_pklot2 + 20)
                # # TODO: this must be refined later for the dist_reward, for now, just simply sum them up
                # dist_reward = dist_reward1 #+ dist_reward2
                if (dist_to_pklot1 < old_dist_to_pklot1) or (dist_to_pklot2 < old_dist_to_pklot2):
                    dist_reward = 0.5
                else:
                    dist_reward = -0.75

            # 3. Penalty to punish the robot if it moves to much

            reward = dist_reward + translate_cost + rotate_cost
        return reward

    def reset(self):
        # del self.
        self.states = []
        self.actions = []
        self.action_indexs = []
        self.log_probs = []
        self.valid_actions = []
        self.rewards = []
        self.episode_length = 0
        self.robot_states = []

    def save_robot_state(self, robot_state):
        self.robot_states.append(robot_state)

    def save_reward(self, reward):
        self.rewards.append(reward)

    def save_action(self, action):
        self.actions.append(action)

    def save_log_prob(self, log_prob):
        self.log_probs.append(log_prob)

    def save_accumulated_reward(self, accumulated_reward):
        pass

    def print_world_obs(self):
        print(self.world_obs)

    def print_world_pklot(self):
        print(self.world_pklot)

    def print_world_robot(self):
        print(self.world_robot)

    # TODO: this function must be changed when writing code of training, now just randomly choose a valid action
    def act(self, actor_net, robot_state):
        # Queries an action from the actor network, should be called from rollout.
        # input is the state at the current timestep
        # Return the probability of the selected action in the distribution

        # Query the actor network for a mean action
        # state_obs_lot = np.array(robot_state.state)
        # state_next_pos = np.array(robot_state.next_pos)
        # state_hitbox = np.array(robot_state.next_hitbox)

        # state = np.dstack((state_obs_lot, state_next_pos, state_hitbox))
        state = np.array([robot_state.state, robot_state.next_pos, robot_state.next_hitbox])

        # actor_net = actor_net.detach().to(self.device)
        state = torch.tensor(state, dtype=torch.float).unsqueeze(dim=0).detach().to(self.device)
        # torch.unsqueeze(state, dim=)
        action_distribution, _ = actor_net(state, 1)
        # print("action_distribution: ", action_distribution)
        valid_action = []
        #
        # print("shape:", action_distribution.shape)

        # We will firstly set Probability of invalid action to zero
        for i in range(len(action_distribution[0])):
            action_index = [i % 9, i // 9]
            if robot_state.moveValidity(action_index) != 0:
                action_distribution[0][i] = 0
                valid_action.append(0)
            else:
                valid_action.append(1)

            # re-normalize the distribution
        total_probability = action_distribution.sum()
        if total_probability < 1e-10:
            print("valid_action number", sum(valid_action))
            print("total_probability: ", total_probability)

        normalized_probabilities = action_distribution / total_probability

        contains_nan = torch.isnan(normalized_probabilities).any().item()
        if contains_nan:
            print("contains NaN value")
            print("valid_action number", sum(valid_action))
            print("total_probability: ", total_probability)
            print("action_distribution: ", action_distribution)
            print("==================================================================")
            print(robot_state.state, robot_state.next_pos, robot_state.next_hitbox)

        action_distribution = Categorical(normalized_probabilities)

        selected_index = action_distribution.sample()
        log_prob = normalized_probabilities.cpu().detach().numpy()[0][selected_index]

        action_index = [selected_index.item() % 9, selected_index.item() // 9]
        # action_index = torch.tensor(selected_index // 9, selected_index % 9).numpy()

        state = state.cpu().detach().numpy()
        # log_prob = log_prob.cpu().detach().numpy()
        action = selected_index.cpu().detach().numpy()
        return state, action, action_index, log_prob, valid_action

    # Plot the environment for every step
    def plot_env(self, step, episode_index):
        plt.switch_backend('agg')
        plt.cla()
        # TODO: might be modified here
        world_obs = self.world[0] * 255
        world_pklots = self.world[1] * 50
        world_robot = self.world[2] * 180
        whole_world = world_obs + world_pklots + world_robot  # 10, 20, 30 is in order to distinguish them in gray level
        plt.imshow(whole_world, cmap="gray_r")
        plt.axis((0, self.world_size[1], self.world_size[0], 0))
        self.img_save_path = IMG_SAVE_PATH
        img_by_episode_save_path = os.path.join(self.img_save_path, 'episode{}'.format(episode_index))
        if not os.path.exists(img_by_episode_save_path):
            os.makedirs(img_by_episode_save_path)
        plt.savefig('{}/step_{}.png'.format(img_by_episode_save_path, step))


def check_available(target, world):
    # check whether the target(60*60)(could be world_pklot or world_robot) area
    # with such position and direction can be placed in the world
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == 1 and world[i][j] == 1:
                # print("will stuck")
                return False
    # print("not stuck")
    return True


# For now, the action is randomly selected
def select_valid_action(robot_state):
    action = robot_state.sample_action()
    action_validity = robot_state.moveValidity(action)
    if action_validity < 0:
        return select_valid_action(robot_state)
    else:
        return action


if __name__ == "__main__":
    # img_save_path = "test_pictures/"
    env = AutoPark_Env()
    env.init_world()
    act = ActorNet().to("cuda")

    env.run_episode(act, 0,0)
