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

    def __init__(self, world0, goals, diagonal):
        assert (len(world0.shape) == 2 and world0.shape == goals.shape)
        self.state = world0.copy()
        self.pos = pos.copy()
        self.agent_pos = self.scanForAgent()
        self.direction =

    def scanForAgent(self):
        agent_pos = (-1, -1)
        agent_last = (-1, -1)
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if (self.state[i, j] > 0):
                    agent_pos = (i, j)
                    agent_last = (i, j)
                if (self.goals[i, j] > 0):
                    agent_goal = (i, j)

        assert (agent_pos != (-1, -1) and agent_goal != (-1, -1))
        assert (agent_pos == agent_last)
        return agent_pos, agent_last, agent_goal

    def setDirection(self, direction):
        self.direction = direction

    def getPos(self):
        x, y = self.agent_pos
        if self.direction == "0":
            return x, y, "0"
        elif self.direction == "1":
            return x, y, "1"
        elif self.direction == "2":
            return x, y, "2"
        elif self.direction == "3":
            return x, y, "3"
        elif self.direction == "4":
            return x, y, "4"
        elif self.direction == "5":
            return x, y, "5"
        elif self.direction == "6":
            return x, y, "6"
        elif self.direction == "7":
            return x, y, "7"

    # return self.agent_pos

    # def getPastPos(self):
    # return self.agent_past

    # def getGoal(self):
    # return self.agent_goal

    # try to move agent and return the status
    def moveAgent(self, direction):
        ax, ay = self.agent_pos

        # Not moving is always allowed
        if (direction == (0, 0)):
            self.agent_past = self.agent_pos
            return 0

        # Otherwise, let's look at the validity of the move
        dx, dy = direction[0], direction[1]
        if (ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[
            1] or ay + dy < 0):  # out of bounds
            return -1
        if (self.state[ax + dx, ay + dy] < 0):  # collide with static obstacle
            return -2

        # No collision: we can carry out the action
        self.state[ax, ay] = 0
        self.state[ax + dx, ay + dy] = 1
        self.agent_past = self.agent_pos
        self.agent_pos = (ax + dx, ay + dy)
        if self.goals[ax + dx, ay + dy] == 1:  # reached goal
            return 1

        # none of the above
        return 0

    # try to execture action and return whether action was executed or not and why
    # returns:
    #     1: action executed and reached goal
    #     0: action executed
    #    -1: out of bounds
    #    -2: collision with wall
    def act(self, action):
        # 0     1  2  3  4
        # still N  E  S  W
        direction = self.getDir(action)
        moved = self.moveAgent(direction)
        return moved

    def getDir(self, action):
        return dirDict[action]

    def getAction(self, direction):
        return actionDict[direction]
