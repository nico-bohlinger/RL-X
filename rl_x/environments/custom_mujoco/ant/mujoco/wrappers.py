import gymnasium as gym


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
            for done, final_info in zip(info["_final_info"], info["final_info"]):
                if done:   
                    for key, info_value in final_info.items():
                        if key not in keys_to_remove:
                            logging_info.setdefault(key, []).append(info_value)

        return logging_info
    

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]
    

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]


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
