from rl_x.environments.vec_env import DummyVecEnv

from rl_x.environments.custom_interface.prototype.custom_environment import CustomEnvironment
from rl_x.environments.custom_interface.prototype.wrappers import NonblockingVecEnv, RecordEpisodeStatistics, RLXInfo


def create_env(config):
    if config.environment.nr_envs == 1:
        env = DummyVecEnv([lambda: CustomEnvironment(config.environment.ip, config.environment.port)])
    else:
        env = NonblockingVecEnv(
                [lambda i=i: CustomEnvironment(config.environment.ip, config.environment.port + i) for i in range(config.environment.nr_envs)], 
                config.environment.synchronized,
                config.environment.async_threshold, 
                start_method="spawn"
        )
    env.num_envs = config.environment.nr_envs
    env = RecordEpisodeStatistics(env)
    env = RLXInfo(env)
    env.action_space.seed(config.environment.seed)
    env.observation_space.seed(config.environment.seed)
    return env
