import numpy as np
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


    def get_single_action_logit_size(self):
        return self.action_space.nvec[0]


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


# The following wrappers are taken from:
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"


    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3


    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated | truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated | truncated:
            self.env.reset(**kwargs)
        return obs, {}


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated | truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated | truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        assert env.observation_space.dtype is not None, "No dtype specified for the observation space"
        assert env.observation_space.shape is not None, "No shape defined for the observation space"
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)
        self._skip = skip


    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)


    def reward(self, reward):
        return np.sign(float(reward))
