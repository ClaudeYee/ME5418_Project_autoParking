import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchviz import make_dot
import numpy as np
from torch import optim
from parameter import LR_ACTOR, LR_CRITIC

class CNNBlock(nn.Module):
    def __init__(self, in_channel=3, output_dim=512):
        super(CNNBlock, self).__init__()
        # conv -> ReLU
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=160, kernel_size=7),
                                        nn.BatchNorm2d(160),
                                        nn.ReLU(inplace=True))  # [24, 24, 160]
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(in_channels=160, out_channels=320, kernel_size=8),
                                        nn.BatchNorm2d(320),
                                        nn.ReLU(inplace=True))  # [17, 22, 320]
        self.cnn_layer3 = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=64, kernel_size=9),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))  # [9, 9, 64]
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))  # [7, 7, 64]
        # dropout -> fc -> ReLU
        self.fc_dropout_layer5 = nn.Sequential(nn.Dropout(),
                                               nn.Linear(3136, 1024),
                                               nn.ReLU(inplace=True))  # [1024, 1]
        # fc
        self.fc_layer6 = nn.Linear(1024, output_dim)  # [512, 1]

    def forward(self, x, batch_size):
        embedding_output = self.cnn_layer1(x)
        embedding_output = self.cnn_layer2(embedding_output)
        embedding_output = self.cnn_layer3(embedding_output)
        embedding_output = self.cnn_layer4(embedding_output)
        embedding_output = embedding_output.view(batch_size, -1)
        embedding_output = self.fc_dropout_layer5(embedding_output)
        output = self.fc_layer6(embedding_output)

        return output


class ActorNet(nn.Module):
    def __init__(self, in_channel=3, lstm_layers=2, output_dim=512, action_dim=27):
        super(ActorNet, self).__init__()
        # CNNBlock + LSTM
        self.cnn = CNNBlock(in_channel, output_dim)
        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=output_dim, num_layers=lstm_layers, batch_first=True)
        self.policy_fc = nn.Linear(output_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, batch_size):
        embedding_output = self.cnn(x, batch_size)
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

    def forward(self, x, batch_size):
        embedding_output = self.cnn(x, batch_size)
        output, (h_n, c_n) = self.lstm(embedding_output)
        # generate the value of current state
        value = self.value_layer(output)
        return value, (h_n, c_n)


class FullModel(nn.Module):
    def __init__(self, device, in_channel=3, lstm_layers=2, output_dim=512, action_dim=81):
        super(FullModel, self).__init__()
        self.device = device
        self.cnn = CNNBlock(in_channel, output_dim)
        self.actor = ActorNet(in_channel, lstm_layers, output_dim, action_dim).to(device)
        self.critic = CriticNet(in_channel, lstm_layers, output_dim).to(device)

    def forward(self, x, batch_size):
        actor_outcome, (h_np, c_np) = self.actor(x, batch_size)
        critic_outcome, (h_nv, c_nv) = self.critic(x, batch_size)
        return actor_outcome, (h_np, c_np), critic_outcome, (h_nv, c_nv)


loss_function = nn.MSELoss()


class FullModelTester:
    def __init__(self, device):
        self.model = self.load_model(device)

    def load_model(self,device):
        model = FullModel(device)
        model.eval()
        return model

    def test_with_input(self, sample_input, batch_size):
        with torch.no_grad():
            actor_outcome, (h_np, c_np), critic_outcome, (h_nv, c_nv) = self.model(sample_input, batch_size)

        return actor_outcome, critic_outcome

    def check_output_dimensions(self, actor_outcome, critic_outcome, expected_actor_shape, expected_critic_shape):
        actor_outcome_shape = actor_outcome.shape
        critic_outcome_shape = critic_outcome.shape

        if actor_outcome_shape == expected_actor_shape and critic_outcome_shape == expected_critic_shape:
            return True
        else:
            return False


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 32
    channels = 3
    height = 60
    width = 60
    num_actions = 81

    expected_actor_shape = (batch_size, num_actions)
    expected_critic_shape = (batch_size, 1)
    # actor_target = torch.randn(1, 81).to(device)

    actor_target = 1/81 * torch.ones(1, 81).to(device)

    # print("actor_target_shape: ", actor_target.shape)
    # critic_target = torch.randn(1, 1)

    loss_function_actor = nn.MSELoss()
    # loss_function_critic = nn.MSELoss()

    actor = ActorNet().to(device)
    optimizer = optim.Adam(actor.parameters(), lr=LR)

    for i in range(1000):
        sample_input = torch.randn(batch_size, channels, height, width, requires_grad=True).to(device)
        print("sample_input_mean: ", sample_input.mean())
        # print("input_shape", sample_input.shape)
        actor_outcome, _ = actor(sample_input, batch_size)

        # actor_outcome, critic_outcome = actor.test_with_input(sample_input,batch_size)
        # print("actor_outcome_shape: ", actor_outcome.shape)
        # print("actor_outcome_type: ", actor_outcome.dtype)

        loss_actor = loss_function_actor(actor_outcome, actor_target)
        print("loss_actor: ", loss_actor)
        # make_dot(loss_actor.mean())
        # loss_actor.requires_grad = True
        # loss_critic = loss_function_critic(critic_outcome, critic_target)
        optimizer.zero_grad()
        loss_actor.backward(retain_graph=True)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5, norm_type=2)
        optimizer.step()
        # loss_critic.backward()

    # is_grad_correct_actor = gradcheck(loss_function_actor)
    # print(is_grad_correct_actor)


    print("Actor Outcome: ", actor_outcome)
    print("Actor Outcome Shape: ", actor_outcome.shape)
    print("\n")
    print("Critic Outcome: ", critic_outcome)
    print("Critic Outcome Shape: ", critic_outcome.shape)
    print("\n")

    # result = tester.check_output_dimensions(actor_outcome, critic_outcome, expected_actor_shape, expected_critic_shape)

    # if result:
    #     print("Model passed the dimension check")
    # else:
    #     print("Model output dimensions do not match expectations")