from ppo_agent import Agent
from Auto_Parking import AutoPark_Env
from model import ActorNet, CriticNet

import torch
import sys
from parameter import *


def train(env, actor_model, critic_model):
    print("Training starts.")
    agent = Agent(env)

    if actor_model != " " and critic_model != " ":
        print(f"Loading in {actor_model} and {critic_model} ...", flush=True)
        agent.actor_net.load_state_dict(torch.load(actor_model))
        agent.critic_net.load_state_dict(torch.load(critic_model))
        print("Model loaded successfully.", flush=True)
    elif actor_model != " " or critic_model != " ":
        print("Errors! Training doesn't start")
        sys.exit(0)
    else:
        print(f"Training from the beginning.", flush=True)

    agent.learn(2000)


def test():
    pass


def main():
    env = AutoPark_Env()
    if TRAIN:
        train(env)
    else:
        test()