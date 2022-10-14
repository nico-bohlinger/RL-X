import envpool

from rl_x.environments.envpool.humanoid_v4.wrappers import RecordEpisodeStatistics


def create_env(config):
    env = envpool.make("Humanoid-v4", env_type="gym", seed=config.algorithm.seed, num_envs=config.algorithm.nr_envs)
    env.num_envs = config.algorithm.nr_envs
    env = RecordEpisodeStatistics(env)
    env.action_space.seed(config.algorithm.seed)
    env.observation_space.seed(config.algorithm.seed)
    return env