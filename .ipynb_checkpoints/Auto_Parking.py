import gym
from gym import spaces
import numpy as np
from collections import OrderedDict
import sys
from matplotlib.colors import hsv_to_rgb
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
    '''
    State.
    Implemented as 2 2d numpy arrays.
    first one "state":
        static obstacle: -1
        empty: 0
        agent = positive integer (agent_id)
    second one "goals":
        agent goal = positive int(agent_id)
    '''

    def __init__(self, world0, pos, carSize):
        #assert (len(world0.shape) == 2 and world0.shape == goals.shape)
        self.state = world0.copy()
        self.pos = pos.copy()
        self.agent_pos, self.direction = self.getPos()
        self.Shape1, self.Shape2 = self.getShape(carSize)


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

    # return self.agent_pos

    # def getPastPos(self):
    # return self.agent_past

    # def getGoal(self):
    # return self.agent_goal

    # try to move agent and return the status
    def moveAgent(self, action):
        # action is a list. Its first element is destination,
        # And its second element is desired heading
        destination, heading = action[0], action[1]

        # Not moving is always allowed
        if destination == self.agent_pos and heading == self.direction:
            return 0

        # Otherwise, let's look at the validity of the move
        Hitbox = self.getHitBox(destination, heading)
        is_in_parking_space = []

        for i in range(len(Hitbox)):
            x, y = Hitbox[i][0], Hitbox[i][1]
            if (x >= self.state.shape[0] or x < 0
                    or y >= self.state.shape[1] or y < 0):  # out of bounds
                return -1

            if self.state[x, y] == (-1):  # collide with static obstacle
                return -2

            elif self.state[x, y] != 0:
                is_in_parking_space.append(self.state[x, y])

        # No collision: we can carry out the action
        self.pos[self.agent_pos] = -1
        self.agent_pos = destination
        self.direction = heading
        self.pos[self.agent_pos] = heading
        self.

        # See if every pixel is in the same parking space
        # If so, then our car parked in its space
        if len(is_in_parking_space) == len(Hitbox):
            return int(is_in_parking_space[0])

        # none of the above
        return 0

    # try to execute action and return whether action was executed or not and why
    # returns:
    #     1: action executed and reached goal
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    def act(self, action):
        direction = self.getDir(action)
        moved = self.moveAgent(direction)
        return moved

    # def getDir(self, action):
    #     return dirDict[action]
    #
    # def getAction(self, direction):
    #     return actionDict[direction]

    # carShape is a three element tuple
    # The first one element is the distant from the centre to the front edge of car
    # The second is the distant to rear edge, the third is to left/right edge
    def getShape(self, carShape):
        # Shape1 is the hitbox when car is at [0, 0] with dir = 0
        # Shape2 is the hitbox with dir = 0
        Shape1 = []
        Shape2 = []

        for i in range(-1 * carShape[2], carShape[2] + 1, 1):
            for j in range(-1 * carShape[1], carShape[0] + 1, 1):
                Shape1.append([i, j])

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
                if ((-1 * carShape[1] - 0.5) < rotated_pixel_coords[0] < (carShape[0] + 0.5) and
                    (-1 * carShape[2] - 0.5) < rotated_pixel_coords[1] < (carShape[2] + 0.5)):
                    # if so, add to Shape2
                    Shape2.append([x, y])

        return Shape1, Shape2

    def getHitBox(self, pos, dir):
        agent_pos = pos
        agent_dir = dir

        shift = np.array(agent_pos)
        hitbox = []

        if agent_dir % 2 == 0:
            Shape = self.Shape1
            angle = agent_dir / 2 * np.pi
        else:
            Shape = self.Shape2
            angle = (agent_dir - 1) / 2 * np.pi

        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        rotateMatrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

        for i in range(len(Shape)):
            rotated_pixel = np.array(Shape[i]).dot(rotateMatrix)
            finial_pixel = rotated_pixel + shift
            hitbox.append([int(finial_pixel[0]), int(finial_pixel[1])])

        return hitbox
