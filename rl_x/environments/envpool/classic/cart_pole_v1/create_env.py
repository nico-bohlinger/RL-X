import envpool

from rl_x.environments.envpool.classic.cart_pole_v1.wrappers import RLXInfo
from rl_x.environments.envpool.classic.cart_pole_v1.general_properties import GeneralProperties


def create_env(config):
    env = envpool.make("CartPole-v1", env_type="gymnasium", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    env.action_space = env.action_space
    env.observation_space = env.observation_space
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space
    env.action_space.seed(config.environment.seed)
    env.observation_space.seed(config.environment.seed)

    return env
