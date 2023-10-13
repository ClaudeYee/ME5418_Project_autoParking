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
from gym.envs.classic_control import rendering

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

    def __init__(self, world0, pos, carSize):
            # assert (len(world0.shape) == 2 and world0.shape == goals.shape)
            self.state = world0.copy()
            self.pos = pos.copy()
            self.agent_pos, self.direction = self.getPos()
            self.shape0, self.shape1 = self.getShape(carSize)
            self.hitbox_index = self.getHitBox(self.agent_pos, self.direction)
            self.hitbox = self.renderHitBox()
            self.num_translation_actions = 9
            # 0: Stay, 1: East, 2: Northeast, 3: North, 4: Northwest, 5: West, 6: Southwest, 7: South, 8: Southeast
            self.num_rotation_actions = 9
            # 0: Stay, 1: East, 2: Northeast, 3: North, 4: Northwest, 5: West, 6: Southwest, 7: South, 8: Southeast

            self.action_space = spaces.Tuple((
                spaces.Discrete(self.num_translation_actions),
                spaces.Discrete(self.num_rotation_actions)
            ))
            #define action space two action:translation and rotation

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

    def getPos(self):
        size = [np.size(self.pos, 0), np.size(self.pos, 1)]
        for i in range(size[0]):
            for j in range(size[0]):
                if self.pos[i, j] != -1:
                    return [i, j], self.pos[i, j]

    # try to move agent and return the status
    def moveAgent(self, action):
        # action is a list. Its first element is destination,
        # And its second element is desired next_dir
        next_pos, next_dir = self.get_new_pos_and_rotation_from_action(action)

        # Not moving is always allowed
        if next_pos == self.agent_pos and next_dir == self.direction:
            return 0

        # Otherwise, let's look at the validity of the move
        hitbox_index = self.getHitBox(next_pos, next_dir)
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
        self.pos[self.agent_pos] = -1
        self.agent_pos = next_pos
        self.direction = next_dir
        self.pos[self.agent_pos] = next_dir
        self.hitbox_index = hitbox_index
        self.hitbox = self.renderHitBox()

        # See if every pixel is in the same parking space
        # If so, then our car parked in its space
        if len(is_in_parking_space) == len(hitbox_index):
            return int(is_in_parking_space[0])

        # none of the above
        return 0

    # try to execute action and return whether action was executed or not and why
    # returns:
    #     1: action executed and reached goal
    #     0: action executed
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
        new_pos = (self.agent_pos[0] + translation[0], self.agent_pos[1] + translation[1])
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
    def getShape(self, carShape):
        # shape0 is the hitbox when car is at [0, 0] with dir = 0
        # shape1 is the hitbox with dir = 1
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

        return shape0, shape1

    # Calculate the hitbox
    def getHitBox(self, pos, dir):
        # desired position and direction
        agent_pos = pos
        agent_dir = dir

        shift = np.array(agent_pos)
        hitbox_index = []

        # See if direction is tilted
        # If not, use Shape1
        if agent_dir % 2 == 0:
            shape = self.shape0
            angle = agent_dir / 2 * np.pi
        # Else, use Shape2
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
    
