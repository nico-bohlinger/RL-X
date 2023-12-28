import gymnasium as gym

from rl_x.environments.gym.mujoco.humanoid_v4.wrappers import RLXInfo, RecordEpisodeStatistics


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("Humanoid-v4", render_mode="human" if config.environment.render else None)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    vector_environment_class = gym.vector.SyncVectorEnv if config.environment.nr_envs == 1 else gym.vector.AsyncVectorEnv
    env = vector_environment_class([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    env = RLXInfo(env)

    env.reset(seed=config.environment.seed)

    return env
