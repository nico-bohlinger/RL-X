import gymnasium as gym

from rl_x.environments.custom_interface.prototype.custom_environment import CustomEnvironment
from rl_x.environments.custom_interface.prototype.wrappers import RLXInfo, RecordEpisodeStatistics
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
    
    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    env = vector_environment_class([make_env(config.environment.seed + i, config.environment.port + i) for i in range(config.environment.nr_envs)])
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)

    return env
