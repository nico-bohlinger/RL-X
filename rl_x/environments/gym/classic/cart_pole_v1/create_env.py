import gymnasium as gym

from rl_x.environments.vec_env import SubprocVecEnv, DummyVecEnv
from rl_x.environments.gym.classic.cart_pole_v1.wrappers import RLXInfo


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("CartPole-v1")
            env = gym.wrappers.RecordEpisodeStatistics(env)
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
    env = RLXInfo(env)
    return env
