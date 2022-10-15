import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl_x.environments.gym.humanoid_v3.wrappers import ConvertInfo


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("Humanoid-v3")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    env = SubprocVecEnv([make_env(config.environment.seed + i) for i in range(config.algorithm.nr_envs)])
    env = ConvertInfo(env)
    return env
