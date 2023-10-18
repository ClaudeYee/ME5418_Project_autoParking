import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
import sys
import time
# from matplotlib.colors import hsv_to_rgb
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

ACTION_COST, IDLE_COST, GOAL_REWARD, COLLISION_REWARD = -0.1, -0.2, 1.0, -1.0

# opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
# JOINT = False # True for joint estimation of rewards for closeby agents
# dirDict = {0:(0,0),1:(0,1),2:(1,0),3:(0,-1),4:(-1,0),5:(1,1),6:(1,-1),7:(-1,-1),8:(-1,1)}
# actionDict={v:k for k,v in dirDict.items()}


class State(object):

    def __init__(self, world0, pos, dir, carSize=ROBOT_SIZE):
            # assert (len(world0.shape) == 2 and world0.shape == goals.shape)
            self.state = world0.copy()
            self.pos = pos.copy()
            self.dir = dir
            self.robot_state = self.getState()      # TODO: This might not be needed later.
            self.robot_size = carSize
            self.shape0, self.shape1 = self.getShape(carSize)   # TODO: here do some changes
            self.hitbox_index = self.getHitBox_index(self.robot_state[0], self.robot_state[1])
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
        size = [np.size(self.pos, 0), np.size(self.pos, 1)]
        for i in range(size[0]):
            for j in range(size[0]):
                if self.pos[i, j] != -1:
                    return [[i, j], self.pos[i, j], self.dir]   # return the coordinate of the robot, the value at this position is 1, and dir refers to the current direction of the robot

    # try to move agent and return the status
    def moveAgent(self, action):
        # action is a list. Its first element is destination,
        # And its second element is desired next_dir
        next_pos, next_dir = self.get_new_pos_and_rotation_from_action(action)

        # Not moving is always allowed
        if next_pos == self.robot_state[0] and next_dir == self.robot_state[1]:
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

        # No collision: we can carry out the action
        self.pos[self.robot_state[0]] = -1
        self.robot_state[0] = next_pos
        self.robot_state[1] = next_dir
        self.pos[self.robot_state[0]] = next_dir
        self.hitbox_index = hitbox_index
        self.hitbox = self.renderHitBox()

        # See if every pixel is in the same parking space
        # If so, then our car parked in its space
        if len(is_in_parking_space) == len(hitbox_index):
            return 2 * int(is_in_parking_space[0])

        # none of the above
        return 1

    # try to execute action and return whether action was executed or not and why
    # returns:
    #     2: action executed and reached goal
    #     1: action executed
    #     0: action executed ==> no moving
    #    -1: out of bounds
    #    -2: collision with wall
    def sample_action(self):
    # sampling actions
        return self.action_space.sample()

    def get_action_data(self, action):
        # return data
        return action

    def get_new_pos_and_rotation_from_action(self, action):
        translation = self.translation_directions[action[0]]
        rotation = action[1]
        new_pos = (self.robot_state[0][0] + translation[0], self.robot_state[0][1] + translation[1])
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
    def __init__(self, world0, plot=False, blank_world=False):          # blank_world: there is no robot and any parking lots
        self.world_size = WORLD_SIZE
        self.robot_size = ROBOT_SIZE
        self.parklot_size = PARKLOT_SIZE

        self.world_obs = None
        self.world_pklot = None
        self.world_robot = None
        self.world = None  # world has two channels: world_obs, world_pklot, world_robot. (3, 40, 40)

        self.init_robot_pos = None
        self.init_robot_dir = None

        self.robot_pos = None
        self.robot_dir = None

        self.init_world(world0)
        self.init_robot_state = None  # might have some problems here

        self.reward = 0

        self.done = False


    # randomly generate a world with obstacles and two parking lots. Note that they are separately put into two channels of the same world
    def init_world(self, world0):
        if world0:
            world = world0.copy()
        else:
            self.world_obs = self.init_obstacles(PROB, self.world_size)       # the obstacle channel of the world
            self.world_pklot = self.init_parkinglots(self.world_size, self.parklot_size)
            self.robot_pos, self.robot_dir, self.world_robot = self.init_robot(self.world_obs)

            self.world = np.array([self.world_obs, self.world_pklot, self.robot_pos])

    def init_obstacles(self, size):
        generate_obs(size)

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
            return pklot_world
        else:
            self.init_parkinglots(world_size, parklot_size)

    # Place the robot into the environment in a random position if the position and any grid that the robot takes are not in the girds of obstacles
    def init_robot(self, world):
        # pos_x and pos_y refer to the center point coordinate of the robot
        pos_x, pos_y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
        # randomly generate a heading of the robot
        dir = random.randint(0, 7)

        init_robot_pos = [pos_x, pos_y]
        init_robot_dir = dir
        self.init_robot_state = State(world, init_robot_pos, init_robot_dir)
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

    def step(self, action):

        # Apply the action
        robot_state = self.init_robot_state
        self.reward += ACTION_COST
        action_outcome = robot_state.moveAgent(self, action)

        if action_outcome < 0:
            self.reward += COLLISION_REWARD
            self.reset()

        elif action_outcome == 2:
            self.reward += GOAL_REWARD
            next_pos, next_dir = robot_state.get_new_pos_and_rotation_from_action(self, action)
            self.init_robot_state = State(self.world, next_pos, next_dir)
            self.done = True

        elif action_outcome == 1:
            next_pos, next_dir = robot_state.get_new_pos_and_rotation_from_action(self, action)
            self.init_robot_state = State(self.world, next_pos, next_dir)

        else:
            self.reward += IDLE_COST

    def reset(self):
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

    def plot_env(self):
        pass


def check_available(target, world):  # check whether the target area with such position and direction can be placed in the world
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            if target[i][j] == world[i][j] == 1:
                return False
        break
    return True
