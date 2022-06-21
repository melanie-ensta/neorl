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


#functions that helps to define actions and environement

def set_actions(data_path, building_id, obs_name, value):
    with open(os.path.join(data_path,'buildings_state_action_space.json'), 'r+') as jsonFile:
        data = json.load(jsonFile)
        data[building_id]['actions'][obs_name] = value
        jsonFile.seek(0)
        json.dump(data, jsonFile)
        jsonFile.truncate()


def citylearn(climate_zone=1, n_buildings=1):
    data_path = os.path.join(os.path.dirname(__file__),"data")
    zone_data_path = os.path.join(data_path,"Climate_Zone_"+str(climate_zone))
    building_attributes = os.path.join(zone_data_path, 'building_attributes.json')
    weather_file = os.path.join(zone_data_path, 'weather_data.csv')
    solar_profile = os.path.join(zone_data_path, 'solar_generation_1kW.csv')
    building_state_actions = os.path.join(data_path,'buildings_state_action_space.json')
    building_ids = ['Building_'+str(i) for i in range(1, n_buildings+1)]
    objective_function = ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','quadratic']
    simulation_period_total=(0,8759)
    simulation_period_len = 8759
    env = CityLearn(zone_data_path,
                    building_attributes,
                    weather_file,
                    solar_profile,
                    building_ids,
                    buildings_states_actions = building_state_actions,
                    simulation_period_total=simulation_period_total,
                    simulation_period_len = simulation_period_len,
                    cost_function = objective_function,
                    central_agent = True,
                    verbose = 0)
    return env


def train_(custom_reward,log_kwargs,model_kwargs,seed):
  data_path = os.path.join(os.path.dirname(__file__), "data")
    #define actions and number of buildings
  set_actions(data_path, 'Building_1', 'dhw_storage', True)
  set_actions(data_path, 'Building_1', 'cooling_storage', True)
  env = citylearn(climate_zone=5, n_buildings=1)
  env.set_custom_reward_function(custom_reward)
  save = log_kwargs['save']
  n_eval_episodes= log_kwargs['n_eval_episodes_callback']
  eval_freq=log_kwargs['eval_freq']
  gamma=model_kwargs['gamma']
  policy=model_kwargs['policy_kwargs']
  train_timesteps=model_kwargs['train_timesteps']
  method=model_kwargs['method']
  n_steps=model_kwargs['n_steps']
  batch_size=model_kwargs['batch_size']
  learning_rate=model_kwargs['learning_rate']
  logdir = 'seed_'+str(seed)

  if not os.path.exists(logdir):
    os.makedirs(logdir)


  env.reset()
  callback = TrackExpectedRewardCallback(eval_env = env, eval_freq = 5000, log_dir = logdir, n_eval_episodes= n_eval_episodes)
  model = PPO("MlpPolicy", env, verbose=0, policy_kwargs =policy, learning_rate=learning_rate, gamma = gamma, n_steps = n_steps, batch_size = batch_size, seed=seed)
  model.learn(total_timesteps= train_timesteps,callback=callback)




def plot_profiles(model,env):
    #simulate one episode with the RBC 
    RBC_train(env) 
    net_elec_consumption_RBC=env.net_electric_consumption
    #reset env to simulate with the saved model
    state=env.reset()
    month=state[0]
    done = False
    model=PPO.load(model)
    action = model.predict(state)
    list_reward=[]
    building_net_demand=[]
    list_obs_soc_dhw=[]
    list_obs_soc_cool=[]
    dhw_stored=[]
    cooling_stored=[]
    action_cooling=[]
    action_dhw=[]
    
    capacity_tank_dhw=5.34 * 3
    capacity_tank_cooling=320.17 * 3
    soc_dhw=0
    soc_cooling=0
    diff_soc_dhw=0
    diff_soc_cooling=0
    
    #simulate one episode with the model and store interesting information
    while not done:
        state, reward, done, _ = env.step(action[0])
        list_reward.append(reward)
        action = model.predict(state)
        building_net_demand.append(env.get_buildings_net_electric_demand())
        list_obs_soc_dhw.append(state[24])
        list_obs_soc_cool.append(state[23])
        diff_soc_dhw=(state[24]-soc_dhw)
        diff_soc_cooling=(state[23]-soc_cooling)
        soc_dhw=state[24]
        soc_cooling=state[23]
        dhw_stored.append(diff_soc_dhw*capacity_tank_dhw)
        cooling_stored.append(diff_soc_cooling*capacity_tank_cooling)
        action_cooling.append(action[0][0])
        action_dhw.append(action[0][1])



  ############ plot on different intervals##############
    interval_1 = range(0,150)   ##see difference with and without storage
    interval_2= range(100,150)   ##period winter + excess generation
    interval_3=range(5000,5050)   ##period summer + no excess generation
    interval_4=range(0,8000)      ##full year

    #choose the interval wanted
    interval=interval_2

    #just to get nice plots
    heure=[str((i//24)*100+i%24) for i in range(24*365)]
    x_axis=heure[interval[0]:interval[-1]+1]
    slices=slice(interval[0],interval[-1]+1,2)



#########plots############

    ## comparison electricity_demand for RBC (rule based controller)/rl/no rl
    plt.figure(figsize=(12,8))
    plt.plot(x_axis,net_elec_consumption_RBC[interval],'--')
    plt.plot(x_axis,env.net_electric_consumption[interval])
    plt.plot(x_axis,env.net_electric_consumption_no_storage[interval])
    plt.xticks(rotation=90)
    plt.ylabel('kW')
    plt.legend([
                'P_grid RBC(kW)',
                'P_grid_storage(kW)',
                'P_grid_no_storage (kW)'])
    plt.title('Total consumption from the grid with and without storage')
    plt.show()


    # # ###diff net_electri consumption storage/no_storage
    # plt.plot(env.net_electric_consumption[interval]-env.net_electric_consumption_no_storage[interval])
    # plt.xlabel('time (hours)')
    # plt.ylabel('consumption_storage - consumption_no_storage (kW)')
    # plt.show()

    # # # ####plot differents profiles
    plt.figure(figsize=(12,8))
    plt.plot(x_axis,env.electric_generation[interval])
    plt.plot(x_axis,env.electric_generation[interval]-(env.electric_consumption_cooling[interval]-env.electric_consumption_cooling_storage[interval])-(env.electric_consumption_dhw[interval]-env.electric_consumption_dhw_storage[interval])-env.electric_consumption_appliances[interval])
    plt.xticks(rotation=90)
    plt.xlabel('Month {}'.format(month))
    plt.ylabel('Consumption (kW)')
    plt.legend([
                'P_pv',
                'Excess generation'])
    plt.title("Electric generation PV and consumption from grid")
    plt.show()

    # ## waste of energy
    # plt.figure(figsize=(12,8))
    # plt.plot(x_axis,env.electric_generation[interval]-(env.electric_consumption_cooling[interval]-env.electric_consumption_cooling_storage[interval])-(env.electric_consumption_dhw[interval]-env.electric_consumption_dhw_storage[interval])-env.electric_consumption_appliances[interval]-env.net_electric_consumption[interval]-((cooling_stored)[interval[0]:interval[-1]+1]))
    # plt.title("Energy loss= excess - energy stored")
    # plt.xticks(rotation=90)
    # plt.ylabel("Energy loss (kW)")
    # plt.show()

    # ##action (a>0 if store, a<0 if release energy)
    plt.figure(figsize=(12,8))
    plt.plot(x_axis,action_cooling[interval[0]:interval[-1]+1])
    plt.plot(x_axis,action_dhw[interval[0]:interval[-1]+1])
    plt.legend([
                'action_cooling',
                'action_dhw'])
    plt.xticks(rotation=90)
    plt.xlabel('Month {}'.format(month))
    plt.title("Actions")
    plt.show()

    # ##action translated in energy stored/released energy
    # plt.figure(figsize=(12,8))
    # plt.plot(x_axis,cooling_stored[interval[0]:interval[-1]+1])
    # plt.plot(x_axis,dhw_stored[interval[0]:interval[-1]+1])
    # plt.legend([
    #             'action_cooling',
    #             'action_dhw'])
    # plt.xlabel('Month {}'.format(month))
    # plt.xticks(rotation=90)
    # plt.title("Energy stored")
    # plt.show()


    ##### action with excess generation
    # plt.figure(figsize=(12,8))
    # plt.plot(x_axis,action_cooling[interval[0]:interval[-1]+1])
    # plt.plot(x_axis,action_dhw[interval[0]:interval[-1]+1])
    # plt.plot(x_axis,env.electric_generation[interval]-(env.electric_consumption_cooling[interval]-env.electric_consumption_cooling_storage[interval])-(env.electric_consumption_dhw[interval]-env.electric_consumption_dhw_storage[interval])-env.electric_consumption_appliances[interval]-env.net_electric_consumption[interval])
    # plt.xlabel('Month {} '.format(month))
    # plt.ylabel('Consumption (kW)')
    # plt.legend([
                
    #             'action_cooling',
    #             'action_dhw',
    #             'P_generation_excess'
    #           ])
    # plt.xticks(rotation=90)
    # plt.title("Excess generation")
    # plt.show()

    #### plots sum of profiles
    # plt.figure(figsize=(12,8))
    # plt.plot(x_axis,env.net_electric_consumption_no_storage[interval])
    # plt.plot(x_axis,env.net_electric_consumption[interval])
    # # plt.plot(env.electric_consumption_dhw[interval]+env.electric_consumption_cooling[interval]+env.electric_consumption_appliances[interval]+env.net_electric_consumption_no_storage[interval]-env.electric_generation[interval]-env.electric_consumption_appliances[interval]-env.)
    # plt.xlabel('Month {}'.format(month))
    # plt.ylabel('kW')
    # plt.legend([
    #             'P_grid',
    #             'P_grid_no_consumption'])
    # plt.xticks(rotation=90)
    # plt.title('Energy supply, and building consumption')
    # plt.show()


    # #### demand for heating and cooling
    # plt.figure(figsize=(12,8))
    # plt.plot(x_axis,env.electric_consumption_cooling[interval]-env.electric_consumption_cooling_storage[interval])
    # plt.plot(x_axis,env.electric_consumption_dhw[interval]-env.electric_consumption_dhw_storage[interval])  
    # plt.xlabel('Month {}'.format(month))
    # plt.ylabel('demand (kW)')
    # plt.legend([
    #             'P_cooling_demand',
    #             'P_dhw_demand'])
    # plt.xticks(rotation=90)
    # plt.title("dhw and cooling demand from the consumers")
    # plt.show()

    #### Action : electricity charge or discharge for cooling and dhw storage
    
    # plt.plot(action_cooling[interval[0]:interval[-1]+1])
    # plt.plot(action_dhw[interval[0]:interval[-1]+1])
    # plt.plot(env.electric_generation[interval])
    # plt.plot(net_electric_consumption_positive[interval])
    # plt.legend([
    #             'action_storage_cooling',
    #             'action_storage_dhw',
    #             'P_pv',
    #             'P_grid'])
    # plt.xlabel('Month {}'.format(month))
    # plt.ylabel(' (kW)')
    # plt.title('Electric generation and action asked by controller')
    # plt.show()

    # ### SOC 
    plt.plot(x_axis,list_obs_soc_cool[interval[0]:interval[-1]+1])
    plt.plot(x_axis,list_obs_soc_dhw[interval[0]:interval[-1]+1])
    plt.legend([
                'SOC_cool',
                'SOC_dhw'])
    plt.xlabel('Month {}'.format(month))
    plt.ylabel('SOC')
    plt.xticks(rotation=90)
    plt.title('SOC of storage devices')
    plt.show()



    #### plot heating+cooling with and without storage

    # plt.plot(interval,env.net_electric_consumption[interval]+env.electric_generation[interval]-env.electric_consumption_appliances[interval])
    # plt.plot(interval,env.net_electric_consumption_no_storage[interval]+env.electric_generation[interval]-env.electric_consumption_appliances[interval])
    # plt.legend([
    #             'P_cooling+heating with storage',
    #             'P_cooling+heating no storage'])
    # plt.xlabel('Month {}'.format(month))
    # plt.ylabel('kW')
    # plt.title('Heating+cooling consumption with and without storage')
    # plt.show()

    ###

    ##### gives the metrics : goal is to have all the metrics <1
    print('The environment cost compared to the RBC is {}'.format(env.cost()))