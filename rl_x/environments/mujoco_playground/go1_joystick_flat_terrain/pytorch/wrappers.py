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
        _, observation = self.env.reset_with_critic_obs()
        info = {}
        return observation, info
    

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        observation = info["observations"]["critic"]
        done = done > 0.5
        truncated = info["time_outs"] > 0.5
        terminated = done & (~truncated)
        return observation, reward, terminated, truncated, info
    

    def get_logging_info_dict(self, info):
        logging_info = {
            key: [info["log"][key],] for key in list(info["log"].keys())
        }

        return logging_info
    

    def close(self):
        pass
