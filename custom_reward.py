import matplotlib.pyplot as plt
import json
import numpy as np
import os

## different reward function defined with electricity_demand


def custom_reward_1(electricity_demand,hour,action,excess):
    return np.array(electricity_demand).sum() **2 *(-1)  if np.array(electricity_demand).sum() < 0 else 0 #r2

def custom_reward_2(electricity_demand,hour,action,excess):
    return np.array(electricity_demand).sum() **3  if np.array(electricity_demand).sum() < 0 else 0 #r2

def custom_reward_3(electricity_demand,hour,action,excess):
    return np.array(electricity_demand).sum()  if np.array(electricity_demand).sum() < 0 else 0

def custom_reward_4(electricity_demand,hour,action,excess):
    if (hour>22 or hour<8 and action[0]>0 and action[1]>0): #good to store by night
      rnight=10000
    else:
      rnight=0
    if (6<hour<10 or 19<hour<22 and action[0]<0 and action[1]<0): #good to release during peak hours
      rday=10000
      
    else:
      rday=0
    elec=np.array(electricity_demand).sum() **3  if np.array(electricity_demand).sum() < 0 else 0
    return(elec+rnight+rday)


def custom_reward_5(electricity_demand,hour,action,excess):
    if (excess>0 and action[0]>0 and action[1]>0):
      rexcess=10000
    else:
      rexcess=0
    elec=np.array(electricity_demand).sum() **3  if np.array(electricity_demand).sum() < 0 else 0
    return(elec+rexcess)


def custom_reward_6(electricity_demand,hour,action,excess):
    if (hour>22 or hour<8 and action[0]>0 and action[1]>0):       
      rnight=1000
    else:
      rnight=0
    if (excess>0 and action[0]>0 and action[1]>0):
      rexcess=1000
    else:
      rexcess=0
    elec=np.array(electricity_demand).sum() **2*(-1)  if np.array(electricity_demand).sum() < 0 else 0
    return(elec+rnight+rexcess)

    

