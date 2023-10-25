import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# class ActorNet(nn.Module):
#     class obs_robot_world1(nn.Module):
#
#         def __init__(self, input_depth, output_depth=None):
#             super(obs_robot_world2, self).__init__()
#
#             if output_depth is None:
#                 output_depth = input_depth * 16
#
#             self.obs_robot_world_input = nn.Sequential(
#                 nn.Conv2d(input_depth, output_depth, 5, stride=1, padding=2),  # 60*60*32
#                 nn.ReLU(),
#                 nn.Conv2d(output_depth, output_depth, 3, stride=1, padding=1),  # 60*60*32
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),  # 30*30*32
#             )
#
#         def forward(self, x):
#             return self.obs_robot_world_input(x)
#
#     class obs_robot_world2(nn.Module):
#
#         def __init__(self, input_depth):
#             super(obs_robot_world2, self).__init__()
#             output_depth = input_depth * 2
#
#             self.obs_robot_world_input = nn.Sequential(
#                 nn.Conv2d(input_depth, output_depth, 3, stride=1, padding=1),  # 30*30*64
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),  # 15*15*64
#                 nn.Conv2d(output_depth, output_depth * 2, 3, stride=2, padding=0),  # 7*7*128
#                 nn.ReLU(),
#                 nn.Conv2d(output_depth * 2, output_depth * 4, 3, stride=2, padding=1),  # 4*4*256
#                 nn.ReLU(),
#                 nn.MaxPool2d(2),  # 2*2*256
#                 nn.Conv2d(output_depth * 4, output_depth * 8, 3, stride=1, padding=1),  # 2*2*512
#                 nn.MaxPool2d(2),  # 512*1
#             )
#
#         def forward(self, x):
#             return self.obs_robot_world2(x)
#
#         class obs_robot_world2(nn.Module):
#
#             def __init__(self, input_depth):
#                 super(obs_robot_world2, self).__init__()
#                 output_depth = input_depth * 2
#
#                 self.obs_robot_world_input = nn.Sequential(
#                     nn.Conv2d(input_depth, output_depth, 3, stride=1, padding=1),  # 30*30*64
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),  # 15*15*64
#                     nn.Conv2d(output_depth, output_depth * 2, 3, stride=2, padding=0),  # 7*7*128
#                     nn.ReLU(),
#                     nn.Conv2d(output_depth * 2, output_depth * 4, 3, stride=2, padding=1),  # 4*4*256
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),  # 2*2*256
#                     nn.Conv2d(output_depth * 4, output_depth * 8, 3, stride=1, padding=1),  # 2*2*512
#                     nn.MaxPool2d(2),  # 1*1*512
#                     nn.Flatten(),  # 512*1
#                 )
#
#             def forward(self, x):
#                 return self.obs_robot_world2(x)
#
#         class pklot_world(nn.Module):
#
#             def __init__(self):
#                 super(pklot_world, self).__init__()
#                 output_depth = 16
#
#                 self.obs_robot_world_input = nn.Sequential(
#                     nn.Conv2d(1, 4, 7, stride=3, padding=2),  # 20*20*4
#                     nn.ReLU(),
#                     nn.MaxPool2d(4),  # 5*5*4
#                     nn.Conv2d(4, output_depth, 3, stride=1, padding=1),  # 5*5*16
#                     nn.ReLU(),
#                     nn.MaxPool2d(5),  # 1*1*16
#                     nn.Flatten(),  # 16*1
#                     nn.Linear(16, 16),  # 16*1
#                     nn.ReLU(),
#
#                 )
#
#             def forward(self, x):
#                 return self.pklot_world(x)

class CNNBlock(nn.Module):
    def __init__(self, in_channel=3, output_dim=512):
        super(CNNBlock, self).__init__()
        # conv -> ReLU
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=160, kernel_size=7),
                                        nn.BatchNorm2d(160),
                                        nn.ReLU(inplace=True))      # [54, 54, 160]
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=320, kernel_size=23),
                                        nn.BatchNorm2d(320),
                                        nn.ReLU(inplace=True))      # [32, 32, 320]
        self.cnn_layer3 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=64, kernel_size=24),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))      # [9, 9, 64]
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))      # [7, 7, 64]
        # dropout -> fc -> ReLU
        self.fc_dropout_layer5 = nn.Sequential(nn.Dropout2d(),
                                               nn.Linear(3136, 1024),
                                               nn.ReLU(inplace=True))   # [1024, 1]
        # fc
        self.fc_layer6 = nn.Linear(1024, output_dim)     # [512. 1]

    def forward(self, x):
        embedding_output = self.cnn_layer1(x)
        embedding_output = self.cnn_layer2(embedding_output)
        embedding_output = self.cnn_layer3(embedding_output)
        embedding_output = self.cnn_layer4(embedding_output)
        embedding_output = self.fc_dropout_layer5(embedding_output)
        output = self.fc_layer6(embedding_output)
        return output


class ActorNet(nn.Module):
    def __init__(self, in_channel=3, lstm_layers=2, output_dim=512, action_dim=18):
        super(ActorNet, self).__init__()
        # CNNBlock + LSTM
        self.cnn = CNNBlock(in_channel, output_dim)
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=output_dim, num_layers=lstm_layers, batch_first=True)
        self.policy_fc = nn.Linear(output_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedding_output = self.cnn(x)
        output, (h_n, c_n) = self.lstm(embedding_output)
        # generate policy distribution on actions
        policy_dist = self.policy_fc(output)
        policy = self.softmax(policy_dist)
        return policy, (h_n, c_n)


class CriticNet(nn.Module):
    def __init__(self, in_channel=3, lstm_layers=2, output_dim=512):
        super(CriticNet, self).__init__()
        self.cnn = CNNBlock(in_channel, output_dim)
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=output_dim, num_layers=lstm_layers, batch_first=True)
        self.value_layer = nn.Linear(output_dim, 1)

    def forward(self, x):
        embedding_output = self.cnn(x)
        output, (h_n, c_n) = self.lstm(embedding_output)
        # generate the value of current state
        value = self.value_layer(output)
        return value, (h_n, c_n)


if __name__ == "__main__":
    actor_net = ActorNet()
    critic_net = CriticNet()
    