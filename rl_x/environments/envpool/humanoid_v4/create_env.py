import envpool

from rl_x.environments.envpool.humanoid_v4.wrappers import RecordEpisodeStatistics, RLXInfo


def create_env(config):
    env = envpool.make("Humanoid-v4", env_type="gym", seed=config.environment.seed, num_envs=config.algorithm.nr_envs)
    env.num_envs = config.environment.nr_envs
    env = RecordEpisodeStatistics(env, config.environment.dones_at_timeout)
    env = RLXInfo(env)
    env.action_space = env.action_space
    env.observation_space = env.observation_space
    env.action_space.seed(config.environment.seed)
    env.observation_space.seed(config.environment.seed)
    return env
