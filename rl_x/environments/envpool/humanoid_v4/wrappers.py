import gym
import numpy as np

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


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
                observations[i] = self.env.reset(np.array([i]))
        return (observations, rewards, dones, infos)
    

class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)


    def get_episode_infos(self, info):
        episode_infos = []
        for maybe_episode_info in info["episode"]:
            if maybe_episode_info is not None:
                episode_infos.append(maybe_episode_info)
        return episode_infos
    

    def get_terminal_observation(self, info, id):
        return None
    

    def get_action_space_type(self):
        return ActionSpaceType.CONTINUOUS

    
    def get_single_action_space_shape(self):
        return self.action_space.shape
    

    def get_observation_space_type(self):
        return ObservationSpaceType.FLAT_VALUES
