from rl_x.environments.environment import Environment
from rl_x.algorithms.algorithm import Algorithm

from rl_x.environments.envpool.humanoid_v4.create_env import create_env as humanoid_v4_create_env
from rl_x.environments.envpool.humanoid_v4.default_config import get_config as humanoid_v4_get_config


class EnvironmentManager:
    def get_default_config(algorithm: Algorithm, environment: Environment):
        if environment == Environment.ENVPOOL_HUMANOID_V4:
            get_config = humanoid_v4_get_config
        else:
            raise NotImplementedError
            
        return get_config(algorithm, environment)


    def get_create_env(environment: Environment):
        if environment == Environment.ENVPOOL_HUMANOID_V4:
            return humanoid_v4_create_env
        else:
            raise NotImplementedError
