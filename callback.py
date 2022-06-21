from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

def evaluate_policy(model, env, n_eval_episodes = 3):
    ep_rewards = []


    for _ in range(n_eval_episodes):
        ep_reward = 0
        obs = env.reset()
        for i in range(8758):
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        ep_rewards.append(ep_reward)


    return(round(np.mean(ep_rewards), 2))

class TrackExpectedRewardCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, log_dir: str, n_eval_episodes: int, verbose=1):
        super(TrackExpectedRewardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.outputfile = None
        self.n_eval_episodes = n_eval_episodes

        self.timesteps = []
        self.expected_rewards = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.outputfile = open(self.log_dir + '/monitoring.txt', 'w')


    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:  ##apply evaluate policy tous les eval_frequency
            expected_reward= evaluate_policy(self.model, self.eval_env, self.n_eval_episodes)
            self.timesteps.append(self.n_calls)
            self.expected_rewards.append(expected_reward)

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.outputfile = open(self.log_dir + '/monitoring.txt', 'w')
        for i in range(len(self.timesteps)):
            self.outputfile.write('{} {}\n'.format(self.timesteps[i], self.expected_rewards[i]))
        self.outputfile.close()
