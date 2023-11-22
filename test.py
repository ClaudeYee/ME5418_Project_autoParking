import os.path
import torch

from Auto_Parking_test import AutoPark_Env
from ppo_agent_test import Agent
from parameter import *

def test():
    env = AutoPark_Env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env, device=device)

    max_testing_timesteps = 1e3
    t = 0

    run_num_pretrained = 1
    buffer_num_saved = 1

    checkpoint_path_actor = ("trained_models" + '/' +"PPO_model_{}".format(run_num_pretrained)
                      + '/' + "PPO_actor_model_{}.pth".format(buffer_num_saved))
    checkpoint_path_critic = ("trained_models" + '/' +"PPO_model_{}".format(run_num_pretrained)
                      + '/' + "PPO_critic_model_{}.pth".format(buffer_num_saved))

    agent.load(checkpoint_path_actor, checkpoint_path_critic)
    print("Successfully load the models!")

    test_running_reward = []

    while t < max_testing_timesteps:
        agent.env.init_world()
        t = agent.env.run_episode(agent.actor_net, t, 0)
        discounted_reward = 0
        for reward in reversed(agent.env.rewards):
            discounted_reward = reward + discounted_reward * GAMMA

        test_running_reward.append(discounted_reward)

    print("============================================================================================")

    avg_test_reward = sum(test_running_reward) / len(test_running_reward)
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

if __name__ == "__main__":
    test()