import gymnasium as gym
import numpy as np
import torch


class RLXInfo(gym.Wrapper):
    def __init__(self, env, nr_envs):
        super(RLXInfo, self).__init__(env)
        self.nr_envs = nr_envs

        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.num_actions,), dtype=np.float32)
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=env.num_privileged_obs, dtype=np.float32)

        # This works because the policy observations are contained at the start of the privileged observations
        self.policy_observation_indices = np.arange(env.num_obs[0])
        self.critic_observation_indices = np.arange(env.num_privileged_obs[0])


    def reset(self):
        observation = self.env.reset()
        info = {}
        return observation, info
    

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        done = done > 0.5
        truncated = info["time_outs"] > 0.5
        terminated = done & (~truncated)
        return observation, reward, terminated, truncated, info
    

    def get_logging_info_dict(self, info):
        print(info)
        exit()
        keys_to_remove = ["final_observation", "_final_observation", "final_info", "_final_info", "elapsed_steps", "_elapsed_steps", "reconfigure", "fail", "episode"]

        if "final_info" in info:
            for key in info["final_info"].keys():
                if key not in keys_to_remove:
                    info[key] = torch.where(info["_final_info"].unsqueeze(1), info["final_info"][key], info[key])
            info["episode_return"] = info["final_info"]["episode"]["return"]
            info["episode_length"] = info["final_info"]["episode"]["episode_len"]

        logging_info = {
            key: info[key].tolist() for key in list(info.keys()) if key not in keys_to_remove
        }

        return logging_info
    

    def close(self):
        pass
