import gymnasium as gym

from rl_x.environments.custom_interface.prototype.custom_environment import CustomEnvironment
from rl_x.environments.custom_interface.prototype.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.custom_interface.prototype.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.custom_interface.prototype.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed, port):
        def thunk():
            env = CustomEnvironment(config.environment.ip, port)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    make_env_functions = [make_env(config.environment.seed + i, config.environment.port + i) for i in range(config.environment.nr_envs)]
    if config.environment.nr_envs == 1:
        env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    return env
