import gymnasium as gym
import numpy as np
    

class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
        self.nr_envs = len(env.all_env_ids)
        self.episode_returns = None


    def reset(self, **kwargs):
        self.episode_returns = np.zeros(self.nr_envs, dtype=np.float32)
        return self.env.reset()


    def step(self, action):
        observations, rewards, terminations, truncations, infos = super(RLXInfo, self).step(action)
        dones = terminations | truncations
        self.episode_returns += rewards
        if any(dones):
            infos["final_info"] = [{} for _ in range(self.nr_envs)]
            infos["final_observation"] = np.zeros_like(observations)
            infos["done"] = dones
            for i in range(len(dones)):
                if dones[i]:
                    for key in infos:
                        keys_to_ignore = ["final_info", "final_observation", "done", "players"]
                        if key not in keys_to_ignore:
                            infos["final_info"][i][key] = np.array(infos[key][i])
                    infos["final_info"][i]["episode_return"] = np.array([self.episode_returns[i]])
                    infos["final_info"][i]["episode_length"] = np.array([infos["elapsed_step"][i]])
                    infos["final_observation"][i] = observations[i].copy()
                    self.episode_returns[i] = 0
                    observations[i], new_info = self.env.reset(np.array([i]))
                    for key in new_info:
                        infos[key][i] = new_info[key]

        return (observations, rewards, terminations, truncations, infos)


    def get_logging_info_dict(self, info):
        all_keys = list(info.keys())
        keys_to_remove = ["final_observation", "final_info", "done", "env_id", "players", "elapsed_step"]

        logging_info = {key: info[key].tolist() for key in all_keys if key not in keys_to_remove}
        if "final_info" in info:
            for done, final_info in zip(info["done"], info["final_info"]):
                if done:   
                    for key, info_value in final_info.items():
                        if key not in keys_to_remove:
                            logging_info.setdefault(key, []).append(info_value)

        return logging_info


    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]
    

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]
    

    def get_single_action_logit_size(self):
        return self.action_space.n
