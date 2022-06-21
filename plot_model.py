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




if __name__ == "__main__":
    #define environment
    data_path = os.path.join(os.path.dirname(__file__), "data")
    set_actions(data_path, 'Building_1', 'dhw_storage', True)
    set_actions(data_path, 'Building_1', 'cooling_storage', True)
    env = citylearn(climate_zone=5, n_buildings=1)
    env.set_custom_reward_function(custom_reward_2)


    #train the bests model with the hyperparameters choosen
    models_dir = "model_PPO_nstep_16384/PPO"

    if not os.path.exists(models_dir):
       os.makedirs(models_dir)


    env.reset()
    model = PPO("MlpPolicy", env, verbose=0,  learning_rate=0.005, n_steps=16384, batch_size=24, gamma = 0.9)
    model.learn(total_timesteps= 100000)
    model.save(f"{models_dir}/{1}")


    ##plot the interesting power profiles + metrics 
    plot_profiles(f"{models_dir}/{1}",env)