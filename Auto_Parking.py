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
from obs_world_generator import generate_obs

from parameter import *

'''
    Observation: (position maps of current agent, current goal, obstacles), vector to goal (vx, vy, norm_v)

    Action space: (Tuple)
        	agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
        5:NE, 6:SE, 7:SW, 8:NW}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''

ACTION_COST, TURNING_COST, CLOSER_REWARD, GOAL_REWARD, FINAL_REWARD = -0.1, -0.2, 0.3, 1.0, 40

# opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
# JOINT = False # True for joint estimation of rewards for closeby agents
# dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
# actionDict={v:k for k,v in dirDict.items()}

class State():
    def __init__(self, world0, pos, carSize=ROBOT_SIZE):
            # assert (len(world0.shape) == 2 and world0.shape == goals.shape)
            self.state = world0.copy()
            self.current_pos = pos
            self.current_dir = dir
            self.next_pos = np.zeros(WORLD_SIZE)
            self.robot_current_state = self.getState()      # TODO: This might not be needed later.
            self.robot_next_state = [[0, 0], 0]
            self.robot_size = carSize
            self.shape0, self.shape1 = self.getShape(carSize)   # TODO: here do some changes
            self.hitbox_index = self.getHitBox_index(self.robot_current_state[0], self.robot_current_state[1])
            self.hitbox = self.renderHitBox()
            self.num_translation_actions = 9
            # 0: Stay, 1: East, 2: Northeast, 3: North, 4: Northwest, 5: West, 6: Southwest, 7: South, 8: Southeast
            self.num_rotation_actions = 9
            # 0: Stay, 1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315 (degree)

            self.action_space = spaces.Tuple((
                spaces.Discrete(self.num_translation_actions),
                spaces.Discrete(self.num_rotation_actions)
            ))
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

    def getState(self):
        size = [np.size(self.current_pos, 0), np.size(self.current_pos, 1)]
        for i in range(size[0]):
            for j in range(size[0]):
                if self.current_pos[i, j] != -1:
                    return [[i, j], self.current_pos[i, j]]   # return the coordinate of the robot, the value at this position is 1, and dir refers to the current direction of the robot

    # try to move agent and return the status
    def moveValidity(self, action):
        # action is a list. Its first element is destination,
        # And its second element is desired next_dir
        next_pos, next_dir = self.get_new_pos_and_rotation_from_action(action)

        # Not moving is always allowed
        if next_pos == self.robot_current_state[0] and next_dir == self.robot_current_state[1]:
            return 0

        # Otherwise, let's look at the validity of the move
        hitbox_index = self.getHitBox_index(next_pos, next_dir)
        is_in_parking_space = []

        for i in range(len(hitbox_index)):
            x, y = hitbox_index[i][0], hitbox_index[i][1]
            if (x >= self.state.shape[0] or x < 0
                    or y >= self.state.shape[1] or y < 0):  # out of bounds
                return -1

            if self.state[x, y] == (-1):  # collide with static obstacle
                return -2

            elif self.state[x, y] != 0:
                is_in_parking_space.append(self.state[x, y])



        # See if every pixel is in the same parking space
        # If so, then our car parked in its space
        if len(is_in_parking_space) == len(hitbox_index):
            return int(is_in_parking_space[0])

        # none of the above
        return 1

    def moveAgent(self, action):
        next_pos, next_dir = self.get_new_pos_and_rotation_from_action(action)

        # refresh robot_current_state & current_pos
        self.robot_current_state = self.robot_next_state.copy()
        self.current_pos = self.next_pos.copy()

        # Valid move: we can carry out the action in next_pos & robot_state
        self.next_pos[self.robot_next_state[0]] = -1
        self.robot_next_state[0] = next_pos
        self.robot_next_state[1] = next_dir
        self.next_pos[self.robot_next_state[0]] = next_dir

        #
        self.hitbox_index = self.getHitBox_index(self.robot_current_state[0], self.robot_current_state[1])
        self.hitbox = self.renderHitBox()

    def sample_action(self):
    # sampling actions
        return self.action_space.sample()

    def get_action_data(self, action):
        # return data
        return action

    def get_new_pos_and_rotation_from_action(self, action):
        translation = self.translation_directions[action[0]]
        rotation = action[1]
        new_pos = (self.robot_current_state[0][0] + translation[0], self.robot_current_state[0][1] + translation[1])
        return new_pos, rotation

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
        carShape = [int((carSize[0]-1)/2), int((carSize[1]-1)/2), int((carSize[1]-1)/2)]
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
                if ((-1 * carShape[1]) < rotated_pixel_coords[0] < (carShape[0]) and
                    (-1 * carShape[2]) < rotated_pixel_coords[1] < (carShape[2])):
                    # if so, add to shape1
                    shape1.append([x, y])
        # TODO: I am not sure why there are two outputs
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
            angle = agent_dir / 2 * np.pi
        # Else, use Shape1
        else:
            shape = self.shape1
            angle = (agent_dir - 1) / 2 * np.pi

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
            hitbox_index.append([int(finial_pixels[0]), int(finial_pixels[1])])

        return hitbox_index

    def renderHitBox(self):
        hitbox = np.zeros([self.state.shape[0], self.state.shape[1]])
        for i in range(len(self.hitbox_index)):
            index0 = self.hitbox_index[i][0]
            index1 = self.hitbox_index[i][1]
            hitbox[index0, index1] = 1
        return hitbox


class AutoPark_Env(gym.Env):
    def __init__(self, world0=None, plot=False, blank_world=False):          # blank_world: there is no robot and any parking lots
        self.world_size = WORLD_SIZE
        self.robot_size = ROBOT_SIZE
        self.parklot_size = PARKLOT_SIZE

        # tunable parameters for dist_reward
        # self.dist_reward_param = DIST_REWARD_PARAM1

        self.world_obs = None
        self.world_pklot = None
        self.world_robot = None
        self.world = None  # world has two channels: world_obs, world_pklot, world_robot. (3, 40, 40)

        self.init_robot_pos = None
        self.init_robot_dir = None

        # self.robot_pos = None
        # self.robot_dir = None

        self.pklot1_pos = None
        self.pklot2_pos = None

        self.init_world(world0)
        self.init_robot_state = None  # might have some problems here

        self.accumulated_reward = 0
        self.episode_length = 200

        self.done = False

    # randomly generate a world with obstacles and two parking lots. Note that they are separately put into two channels of the same world
    def init_world(self, world0):
        if world0:
            world = world0.copy()
        else:
            self.world_obs = self.init_obstacles(self.world_size)       # the obstacle channel of the world
            self.world_pklot = self.init_parkinglots(self.world_size, self.parklot_size)
            self.robot_pos, self.robot_dir, self.world_robot = self.init_robot(self.world_obs)

            self.world = np.array([self.world_obs, self.world_pklot, self.world_robot])

    def init_obstacles(self, size):
        world_obs = generate_obs(size)
        return world_obs

    def init_parkinglots(self, world_size, parklot_size):
        # Create the parking world
        pklot_world = np.zeros(world_size)

        # Generate the two parking lots directions
        # 0: horizontal parking lot and 1: vertical parking lot
        pklot1_dir = random.choice([0, 1])
        pklot2_dir = random.choice([0, 1])

        # From the parking lots size and direction generate two parking lot that will be added to the parking world
        # via arrays of 1 if the parking lot is horizontal or 2 if it is vertical
        def parklot_generate(dir, num, size):
            if dir == 0:
                return num * np.ones(size)
            else:
                return num * np.transpose(np.ones(size))

        pklot1 = parklot_generate(pklot1_dir, 1, parklot_size)
        pklot2 = parklot_generate(pklot2_dir, 2, parklot_size)

        # Add pklot1 to the world
        x1, y1 = np.random.randint(0, pklot_world.shape[0] - pklot1.shape[0] + 1), np.random.randint(0,
                                                                                                     pklot_world.shape[
                                                                                                         1] -
                                                                                                     pklot1.shape[
                                                                                                         1] + 1)
        while np.all(pklot_world[x1:x1 + pklot1.shape[0], y1:y1 + pklot1.shape[1]]) != 0:
            x1, y1 = np.random.randint(0, pklot_world.shape[0] - pklot1.shape[0] + 1), np.random.randint(0,
                                                                                                         pklot_world.shape[
                                                                                                             1] -
                                                                                                         pklot1.shape[
                                                                                                             1] + 1)
        pklot_world[x1:x1 + pklot1.shape[0], y1:y1 + pklot1.shape[1]] = pklot1

        # Randomly choose positions for matrix2, ensuring no overlap with matrix1
        while True:
            x2, y2 = np.random.randint(0, pklot_world.shape[0] - pklot2.shape[0] + 1), np.random.randint(0,
                                                                                                         pklot_world.shape[
                                                                                                             1] -
                                                                                                         pklot2.shape[
                                                                                                             1] + 1)
            if np.all(pklot_world[x2:x2 + pklot2.shape[0], y2:y2 + pklot2.shape[1]] == 0):
                pklot_world[x2:x2 + pklot2.shape[0], y2:y2 + pklot2.shape[1]] = pklot2
                break

        if check_available(pklot1, self.world_obs) and check_available(pklot2, self.world_obs):

            self.pklot1_pos = [(2* x1 + pklot1.shape[0])/2, (2* y1 + pklot1.shape[1])/2]
            self.pklot2_pos = [(2* x2 + pklot2.shape[0])/2, (2* y2 + pklot2.shape[1])/2]

            return pklot_world

        else:
            self.init_parkinglots(world_size, parklot_size)

    # Place the robot into the environment in a random position if the position and any grid that the robot takes are not in the girds of obstacles
    def init_robot(self, world):
        # pos_x and pos_y refer to the center point coordinate of the robot
        pos_x, pos_y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
        # randomly generate a heading of the robot
        dir = random.randint(0, 7)
        init_robot_pos_coord = [pos_x, pos_y]
        init_robot_dir = dir
        init_robot_pos = -1 * np.ones(world.shape[0], world.shape[1])
        init_robot_pos[init_robot_pos_coord[0], init_robot_pos_coord[1]] = init_robot_dir

        self.init_robot_state = State(world, init_robot_pos)
        init_shape = self.init_robot_state.getShape(ROBOT_SIZE)
        init_hitbox = self.init_robot_state.hitbox

        # determine whether the robot center has been placed in the free space
        # TODO: this is defined in robot channel of the whole map (there are other channels)
        if world[pos_x, pos_y] == 0 and check_available(init_hitbox, world):
            init_robot_pos = [pos_x, pos_y]
            init_robot_dir = dir
            init_robot_hitbox = init_hitbox
            return init_robot_pos, init_robot_dir, init_robot_hitbox
        else:
            self.init_robot(world)

    def step(self, robot_state, done=False):
        # If the time step is still not done we can verify if the action is valid and if yes we can complete the action
        # and change the state of our robot and the different parameters accordingly
        ## For now, randomly sample an action from the valid action space for testing without training
        current_robot_state = robot_state
        action = select_valid_action(current_robot_state)
        self.save_action(action)
        # TODO: based on this line, modify moveAgent()
        # robot_state transition
        next_robot_state = robot_state.moveAgent(action)
        next_robot_pos, next_robot_dir = robot_state.get_new_pos_and_rotation_from_action(action)
        # TODO: based on this line, modify the function to return which parking lot it reaches, if the task is complete.
        if next_robot_state.parking_complete() == 1 or next_robot_state.parking_complete() == 2:
            done = True
        # robot receives reward after conducting an action
        reward = self.compute_reward(next_robot_pos, next_robot_dir, done)

        return next_robot_state, reward, done
            # Check the action
            # action_validity = robot_state.moveValidity(self, action)
            # If the move is not valid (ie out of bond or collision) the robot needs to select another action
            # as part of this same step and without getting punished or rewarded
            # if action_validity < 0:
            #     action = robot_state.sample_action()

            # If the move is valid
            # else:
            #     # No matter the outcome of the move these need to be updated in the same way
            #     self.episode_length -= 1
            #     robot_state.moveAgent(self, action)
            #     next_pos, next_dir = robot_state.get_new_pos_and_rotation_from_action(self, action)
            #
            #     # If the action is valid and does not lead to the completion of the mission the robot and consists only
            #     # in the robot turning without moving, the robot complete the turn and is rewarded accordingly
            #     if action_validity == 0:
            #         self.reward += TURNING_COST
            #
            #     # If the action is valid and does not lead to the completion of the mission but the robot move
            #     # it will complete the move and is rewarded accordingly depending on whether it get closer or
            #     # not to one of the parking lot
            #     elif action_validity == 1:
            #         pre_dist1 = np.linalg.norm(self.robot_pos - self.pklot1_pos)
            #         pre_dist2 = np.linalg.norm(self.robot_pos - self.pklot2_pos)
            #
            #         post_dist1 = np.linalg.norm(next_pos - self.pklot1_pos)
            #         post_dist2 = np.linalg.norm(next_pos - self.pklot2_pos)
            #
            #         self.reward += ACTION_COST
            #
            #         # If the robot is getting closer to one of the parking lot after the move
            #         # it is rewarded if not it is punished
            #         if (pre_dist1 > post_dist1) or (pre_dist2 > post_dist2):
            #             self.reward += CLOSER_REWARD
            #         else:
            #             self.reward -= CLOSER_REWARD
            #
            #     # If the action is valid and lead to the completion of the mission the robot move one step and is
            #     # reward accordingly and the mission is completed and the done parameter is changed to True to indicate
            #     # the mission is done and the episode can be reset
            #     else:
            #         self.reward += ACTION_COST
            #         self.reward += GOAL_REWARD
            #         self.done = True
            #
            #     # Update the state after the action is completed and the reward is given
            #     self.init_robot_state = State(self.world, next_pos, next_dir)

        # If the time step is exhausted the episode will be stopped no matter if the mission is completed or not and
        # the environment and episode will both be reset
        # else:
        #     self.done = True

    def run_episode(self):
        done = False
        robot_state = self.init_robot_state # start the first step from the init_robot_state, and set it as the robot_current_state
        self.save_robot_state(robot_state)  # save the initial robot state
        for _ in range(self.episode_length):
            # if the task is not completed or not reach episode_length, do step for state transition and save robot_state and reward
            robot_state, reward, done = self.step(robot_state=robot_state, done=done)
            self.accumulated_reward += reward
            self.save_robot_state(robot_state)
            self.save_reward(reward)
            if done:
                self.save_accumulated_reward(self.accumulated_reward)
                print("The robot has successfully parked in the parking lot, task succeeded!")
                break
        if not done:
            print("The steps in this episode have exceeded the episode length we set, task failed.")

    # TODO: Let's define it later, because for now, the design of reward mechanism is not so important for env testing.
    # TODO: NOTE: This part is extremely important for training the agent!!! It might be revised for many times.
    def compute_reward(self, robot_pos, robot_dir, done):
        # 1. Reward for task completion
        if done:
            final_reward = FINAL_REWARD   # if the robot successfully has parked into the the parking lot, it receives a very big reward
            reward = final_reward
        else:
            # 2. Reward to stimulate the robot to get closer to the parking lots (either parking lot 1 or parking lot 2)
            dist_to_pklot1 = np.linalg.norm(robot_pos - self.pklot1_pos)
            dist_to_pklot2 = np.linalg.norm(robot_pos - self.pklot2_pos)
            # dist_reward_param is a tunable parameter to guarantee tht dist_reward will never greater than the final reward
            dist_reward1 = self.dist_reward_param * (1 / dist_to_pklot1)
            dist_reward2 = self.dist_reward_param * (1 / dist_to_pklot2)
            # TODO: this must be refined later for the dist_reward, for now, just simply sum them up
            dist_reward = dist_reward1 + dist_reward2
            # 3. Penalty to punish the robot if it moves to much
            negative_reward = ACTION_COST
            reward = dist_reward + negative_reward
        return reward

    def save_robot_state(self, robot_state):
        pass

    def save_reward(self, reward):
        pass

    def save_action(self, action):
        pass

    def save_accumulated_reward(self, accumulated_reward):
        pass
    
    def getObstacleWorld(self):
        print(self.world_obs)

    def getParklotWorld(self):
        print(self.world_pklot)

    def getRobotWorld(self):
        print(self.world_robot)



    # TODO: this function must be changed when writing code of training, now just randomly choose a valid action
    def act(self, actions):
        pass

    # Plot the environment for every step
    def plot_env(self, img_save_path, step):
        plt.switch_backend('agg')
        plt.cla()
        # TODO: might be modified here
        whole_world = self.world[0] + 10 * self.world[1] + 20 * self.world[2]       # 10, 20, 30 is in order to distinguish them in gray level
        plt.imshow(whole_world, cmap="gray")
        plt.axis((0, self.world_size[1], self.world_size[0], 0))
        plt.savefig('{}/step_{}.png'.format(img_save_path, step))


def check_available(target, world):  # check whether the target area with such position and direction can be placed in the world
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == world[i][j] == 1:
                return False
        break
    return True


def select_valid_action(robot_state):
    action = robot_state.sample_action()
    action_validity = robot_state.moveValidity(action)
    if action_validity < 0:
        select_valid_action(robot_state)
    else:
        return action


if __name__ == "__main__":
    img_save_path = "test_pictures/"
    if os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    env = AutoPark_Env()