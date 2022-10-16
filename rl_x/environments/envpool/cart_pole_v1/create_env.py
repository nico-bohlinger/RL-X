import envpool

from rl_x.environments.envpool.cart_pole_v1.wrappers import RecordEpisodeStatistics, RLXInfo


def create_env(config):
    env = envpool.make("CartPole-v1", env_type="gym", seed=config.environment.seed, num_envs=config.algorithm.nr_envs)
    env.num_envs = config.algorithm.nr_envs
    env = RecordEpisodeStatistics(env)
    env = RLXInfo(env)
    env.action_space.seed(config.environment.seed)
    env.observation_space.seed(config.environment.seed)
    return env
