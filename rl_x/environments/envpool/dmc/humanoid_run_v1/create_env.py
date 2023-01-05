import numpy as np
import envpool
from gymnasium.spaces import Box

from rl_x.environments.envpool.dmc.humanoid_run_v1.wrappers import RecordEpisodeStatistics, RLXInfo


def create_env(config):
    env = envpool.make("HumanoidRun-v1", env_type="dm", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    env.num_envs = config.environment.nr_envs
    as_spec = env.action_spec()
    env.action_space = Box(low=np.full(as_spec.shape, as_spec.minimum), high=np.full(as_spec.shape, as_spec.maximum), shape=as_spec.shape, dtype=np.float32, seed=config.environment.seed)
    os_shape = (95,)
    env.observation_space = Box(low=np.full(os_shape, 0.0), high=np.full(os_shape, np.inf), shape=os_shape, dtype=np.float32, seed=config.environment.seed)
    env = RecordEpisodeStatistics(env)
    env = RLXInfo(env)
    return env
