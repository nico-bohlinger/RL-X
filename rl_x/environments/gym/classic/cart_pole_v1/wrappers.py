import gymnasium as gym

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
    

    def get_logging_info_dict(self, info):
        all_keys = list(info.keys())
        keys_to_remove = ["final_observation", "final_info"]

        logging_info = {
            key: info[key][info["_" + key]].tolist()
                for key in all_keys if key not in keys_to_remove and not key.startswith("_") and len(info[key][info["_" + key]]) > 0
        }
        if "final_info" in info:
            for is_final_info, final_info in zip(info["_final_info"], info["final_info"]):
                if is_final_info:   
                    for key, info_value in final_info.items():
                        logging_info.setdefault(key, []).append(info_value)

        return logging_info 

    
    def get_action_space_type(self):
        return ActionSpaceType.DISCRETE


    def get_observation_space_type(self):
        return ObservationSpaceType.FLAT_VALUES


    def get_single_action_space_shape(self):
        return ()


    def get_single_observation_space_shape(self):
        return (self.observation_space.shape[1],)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.episode_returns = None
        self.episode_lengths = None


    def reset(self, **kwargs):
        self.episode_return = 0.0
        self.episode_length = 0.0
        return self.env.reset(**kwargs)


    def step(self, action):
        observation, reward, termination, truncation, info = super(RecordEpisodeStatistics, self).step(action)
        done = termination | truncation
        self.episode_return += reward
        self.episode_length += 1
        if done:
            info["episode_return"] = self.episode_return
            info["episode_length"] = self.episode_length
        return (observation, reward, termination, truncation, info)
