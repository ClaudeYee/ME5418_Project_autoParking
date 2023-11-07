from ppo_agent import Agent
from Auto_Parking import AutoPark_Env
from model import ActorNet, CriticNet

import torch
import sys
from parameter import *


def train(agent, actor_model, critic_model):
    print("Training starts.")
    if actor_model != " " and critic_model != " ":
        print(f"Loading in {actor_model} and {critic_model} ...", flush=True)
        agent.actor_net.load_state_dict(torch.load(actor_model))
        agent.critic_net.load_state_dict(torch.load(critic_model))
        print("Model loaded successfully.", flush=True)
    elif actor_model != " " or critic_model != " ":
        print("Error! Training doesn't start")
        sys.exit(0)
    else:
        print(f"Training from the beginning.", flush=True)

    agent.learn(total_timesteps=2000)


def test(agent, actor_model):
    print(f"Testing {actor_model}", flush=True)
    if actor_model == '':
        print('Error! Model file not specified', flush=True)
        sys.exit(0)

    # TODO: the dimensions here might be changed to adjust the dimension defined in our neural network
    state_dim = env.world.shape  # [3, 60, 60]
    action_dim = env.actions.shape  # [1, 81]

    policy = ActorNet(in_channel=IN_CHANNEL, lstm_layers=LSTM_LAYERS,
                      output_dim=OUTPUT_DIM, action_dim=action_dim)
    policy.load_state_dict(torch.load(actor_model))

    # TODO: need to define a function to evaluate the policy or import it form another file/part of our code
    # eval_policy(policy, env, render=True)


def main():
    device = torch.device("cuda") if USE_GPU else torch.device("cpu")

    env = AutoPark_Env()
    agent = Agent()

    if TRAIN:
        train(env)
    else:
        test()

if __name__ == '__main__':
    main()