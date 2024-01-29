import gymnasium as gym

from rl_x.environments.gym.atari.pong_v5.wrappers import RLXInfo, RecordEpisodeStatistics, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv
from rl_x.environments.gym.atari.pong_v5.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.gym.atari.pong_v5.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("ALE/Pong-v5", render_mode="human" if config.environment.render else None)
            env = RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    make_env_functions = [make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)]
    if config.environment.nr_envs == 1:
        env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    env.reset(seed=config.environment.seed)
    
    return env
