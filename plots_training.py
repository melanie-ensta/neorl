import os
import json
import numpy as np
from util import *
import multiprocessing

import matplotlib.pyplot as plt
import json
import numpy as np
import os
import stable_baselines3
from stable_baselines3 import A2C,PPO, TD3, SAC
import torch as th
from neorl.neorl_envs.citylearn.citylearn import CityLearn, __file__
from custom_reward import custom_reward_1, custom_reward_2, custom_reward_3, custom_reward_4, custom_reward_5, custom_reward_6
from RBC import RBC_Agent, RBC_train
from stable_baselines3.common.callbacks import BaseCallback
from callback import *
from stable_baselines3.common.monitor import Monitor
 
 
    #function that plot expected_reward over training period for different parameters
def plot_monitoring(benchmark_learning,file1: str, file2: str, file3: str):
    file1 = open(file1, 'r')
    file2 = open(file2, 'r')
    file3 = open(file3, 'r')
    timesteps = []
    expected_rewards1 = []
    expected_rewards2 = []
    expected_rewards3 = []
    for line1 in file1:
        values1 = line1.split()
        timesteps.append(int(values1[0]))
        expected_rewards1.append(float(values1[1]))
    for line2 in file2:
        values2= line2.split()
        expected_rewards2.append(float(values2[1]))
    for line3 in file3:
        values3= line3.split()
        expected_rewards3.append(float(values3[1]))

    plt.plot(timesteps,[benchmark_learning for x in range(len(timesteps))])
    plt.plot(timesteps, expected_rewards1)
    plt.plot(timesteps, expected_rewards2)
    plt.plot(timesteps, expected_rewards3)
    plt.xlabel('Timesteps')
    plt.ylabel('Mean reward over one episode')
    plt.legend([
                'RBC',
                'n_steps=2048',
                'n_steps=8192',
                'n_steps=16384'])
    plt.show()


if __name__ == "__main__":
    plot_monitoring(-938937696,'Exp_parameters_reward_2/PPO_0.9_200000_24/0.1_2048/seed_1/monitoring.txt','Exp_parameters_reward_2/PPO_0.9_200000_24/0.1_8192/seed_1/monitoring.txt','Exp_parameters_reward_2/PPO_0.9_200000_24/0.1_8192/seed_1/monitoring.txt')