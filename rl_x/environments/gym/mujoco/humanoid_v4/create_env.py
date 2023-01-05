import gymnasium as gym
import numpy as np

from rl_x.environments.vec_env import SubprocVecEnv
from rl_x.environments.gym.mujoco.humanoid_v4.wrappers import RLXInfo


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("Humanoid-v4")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    env = SubprocVecEnv([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    env.seed(config.environment.seed)
    env = RLXInfo(env)
    return env
