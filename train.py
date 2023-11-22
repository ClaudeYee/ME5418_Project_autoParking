import os.path
import torch

from Auto_Parking_test import AutoPark_Env
from ppo_agent_test import Agent
from parameter import *

import time
from datetime import datetime
def train():
    # Initialize hyperparameters

    max_training_timesteps = int(2e4)

    save_model_freq = 16

    checkpoint_dir = "trained_models"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_dir = checkpoint_dir + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    current_num_files = next(os.walk(checkpoint_dir))[1]
    run_num_pretrained = len(current_num_files)

    checkpoint_dir = checkpoint_dir + "PPO_model_{}".format(run_num_pretrained)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    # training procedure
    env = AutoPark_Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device=device)
    agent.learn(max_training_timesteps, checkpoint_dir, save_model_freq)

if __name__ == "__main__":
    train()



