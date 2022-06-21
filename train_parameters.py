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


Learning_rate = {
    '0.1' : 0.1
    # '0.01' : 0.01,
    # '0.001' : 0.001
}

N_steps  = {
    '2048' : 2048,
    '8192' : 8192,
    '16384' : 16384
}

if __name__ == "__main__":

    log_kwargs = {'save' : True, 'n_eval_episodes_callback' : 3, 'eval_freq' : 5000}

    model_kwargs = {
        'gamma' : 0.90,
        'policy_kwargs' : dict(activation_fn = th.nn.LeakyReLU, net_arch = [dict(pi = [256], vf = [256])]),
        'train_timesteps' : 200000,
        'method' : 'PPO',
        'n_steps' : 8192,
        'batch_size' : 24,
        'learning_rate' : 0.1
    }


    seeds = [1, 2]

    custom_reward=custom_reward_2

    if not os.path.exists('Exp_parameters_reward_2'):
        os.mkdir('Exp_parameters_reward_2')
    os.chdir('Exp_parameters_reward_2')
    
    name = model_kwargs['method']+'_'+str(model_kwargs['gamma'])+'_'+str(model_kwargs['train_timesteps'])+'_'+str(model_kwargs['batch_size'])

                        
                        
    if not os.path.exists(name):
        os.mkdir(name)
    os.chdir(name)

    for key1, value1 in Learning_rate.items():
        for key2, value2 in N_steps.items():
            log = key1 + '_' + str(key2)

            os.mkdir(log)
            os.chdir(log)
            #Multi Porcessing
            
            model_kwargs['learning_rate']=value1
            model_kwargs['n_steps'] = value2

            train_(custom_reward=custom_reward,log_kwargs=log_kwargs,model_kwargs=model_kwargs,seed=1)

            os.chdir('../')
                        
    os.chdir('../')