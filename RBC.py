import matplotlib.pyplot as plt
import json
import numpy as np
import os

#Rule-Based controller definition
class RBC_Agent:
    def __init__(self, actions_spaces):
        self.actions_spaces = actions_spaces
        self.reset_action_tracker()
        
    def reset_action_tracker(self):
        self.action_tracker = []
        
    def select_action(self, states):
        hour_day=states[2]
        # Daytime: release stored energy
        a = [[0.0 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        if hour_day >= 9 and hour_day <= 21:
            a = [[-0.08 for _ in range(len(self.actions_spaces[i].sample()))] for i in range(len(self.actions_spaces))]
        
        # Early nightime: store DHW and/or cooling energy
        if (hour_day >= 1 and hour_day <= 8) or (hour_day >= 22 and hour_day <= 24):
            a = []
            for i in range(len(self.actions_spaces)):
                if len(self.actions_spaces[i].sample()) == 2:
                    a.append([0.091, 0.091])
                else:
                    a.append([0.091])
   
        self.action_tracker.append(a)
        return np.array(a)[0]

#RBC performance for a specific environment

def RBC_train(env):
# Instantiating the control agent(s)
    observations_spaces, actions_spaces = env.get_state_action_spaces()
    agents = RBC_Agent(actions_spaces)

    state = env.reset()
    done = False
    rewards_list = []
    state_list=[state[-1]]

    while not done:
        action = agents.select_action(state)
        next_state, rewards, done, _ = env.step(action)
        state = next_state
        rewards_list.append(rewards)
        state_list.append(state[-1])

    benchmark_training=sum(rewards_list)
    return(benchmark_training)


