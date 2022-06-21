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


Network = {
    1 : [dict(pi = [64,64], vf = [64,64])],
    2 : [dict(pi = [8,8], vf = [8,8])],
    3 : [dict(pi = [256], vf = [256])],
}

Activation  = {
    'relu' : th.nn.ReLU,
    'tanh' : th.nn.Tanh,
    'leakyrelu' : th.nn.LeakyReLU
}

if __name__ == "__main__":

    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 3, 'eval_freq' : 5000}

    model_kwargs = {
        'gamma' : 0.90,
        'policy_kwargs' : dict(activation_fn = th.nn.Tanh, net_arch = [dict(pi = [64,64], vf = [64,64])]),
        'train_timesteps' : 200000,
        'method' : 'PPO',
        'n_steps' : 2000,
        'batch_size' : 1024,
        'learning_rate' : 0.005
    }


    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    custom_reward=custom_reward_2

    if not os.path.exists('Exp_network_reward_1'):
        os.mkdir('Exp_network_reward_1')
    os.chdir('Exp_network_reward_1')
    
    name = model_kwargs['method']+'_'+str(model_kwargs['learning_rate'])+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['n_steps'])+'_'+str(model_kwargs['batch_size'])

                        
                        
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for key1, value1 in Activation.items():
        for key2, value2 in Network.items():
            log = key1 + '_' + str(key2)

            os.mkdir(log)
            os.chdir(log)
            #Multi Porcessing
            
            model_kwargs['policy_kwargs'] = dict(activation_fn = value1, net_arch = value2)

            train_(custom_reward=custom_reward,log_kwargs=log_kwargs,model_kwargs=model_kwargs,seed=1)
            # processes = [multiprocessing.Process(target = train_, args = [custom_reward,log_kwargs, model_kwargs, seed]) for seed in seeds]

            # for process in processes:
            #   process.start()
            # for process in processes:
            #   process.join()

            os.chdir('../')
                        
    os.chdir('../')