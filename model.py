import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    class obs_robot_world1(nn.Module):

        def __init__(self, input_depth, output_depth = None):
            super(obs_robot_world2, self).__init__()

            if output_depth is None:
                output_depth = input_depth * 16

            self.obs_robot_world_input = nn.Sequential(
                nn.Conv2d(input_depth, output_depth, 5, stride=1, padding=2), # 60*60*32
                nn.ReLU(),
                nn.Conv2d(output_depth, output_depth, 3, stride=1, padding=1), # 60*60*32
                nn.ReLU(),
                nn.MaxPool2d(2), # 30*30*32
            )

        def forward(self, x):
            return self.obs_robot_world_input(x)

    class obs_robot_world2(nn.Module):

        def __init__(self, input_depth):
            super(obs_robot_world2, self).__init__()
            output_depth = input_depth*2

            self.obs_robot_world_input = nn.Sequential(
                nn.Conv2d(input_depth, output_depth, 3, stride=1, padding=1),   # 30*30*64
                nn.ReLU(),
                nn.MaxPool2d(2),    # 15*15*64
                nn.Conv2d(output_depth, output_depth * 2, 3, stride=2, padding=0),    # 7*7*128
                nn.ReLU(),
                nn.Conv2d(output_depth * 2, output_depth * 4, 3, stride=2, padding=1),  # 4*4*256
                nn.ReLU(),
                nn.MaxPool2d(2),    # 2*2*256
                nn.Conv2d(output_depth * 4, output_depth * 8, 3, stride=1, padding=1),  # 2*2*512
                nn.MaxPool2d(2),  # 512*1
            )

        def forward(self, x):
            return self.obs_robot_world_input(x)


class CriticNet(nn.Module):
    pass
