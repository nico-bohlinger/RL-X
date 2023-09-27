import gymnasium as gym
import numpy as np

from rl_x.environments.vec_env import SubprocVecEnv, DummyVecEnv
from rl_x.environments.custom_mujoco.ant.environment import Ant
from rl_x.environments.custom_mujoco.ant.wrappers import RecordEpisodeStatistics, RLXInfo


def create_env(config):
    if config.environment.render and config.environment.vec_env_type == "subproc":
        raise ValueError("Cannot render with vec_env_type='subproc'. Use vec_env_type='dummy' instead.")
    def make_env(seed):
        def thunk():
            env = Ant(render=config.environment.render)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    if config.environment.vec_env_type == "subproc":
        env = SubprocVecEnv([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    elif config.environment.vec_env_type == "dummy":
        env = DummyVecEnv([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    else:
        raise ValueError("Unknown vec_env_type")
    env.seed(config.environment.seed)
    env = RecordEpisodeStatistics(env)
    env = RLXInfo(env)
    return env
