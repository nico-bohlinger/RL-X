import gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super(RecordEpisodeStatistics, self).step(
            action
        )
        self.episode_returns += rewards
        self.episode_lengths += 1
        infos["episode"] = [None] * self.num_envs
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length
                }
                infos["episode"][i] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (observations, rewards, dones, infos)