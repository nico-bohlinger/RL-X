import gymnasium as gym

from rl_x.environments.vec_env import SubprocVecEnv, DummyVecEnv
from rl_x.environments.gym.atari.pong_v5.wrappers import RLXInfo, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, ClipRewardEnv


def create_env(config):
    def make_env(seed):
        def thunk():
            env = gym.make("ALE/Pong-v5")
            env = gym.wrappers.RecordEpisodeStatistics(env)
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
    if config.environment.vec_env_type == "subproc":
        env = SubprocVecEnv([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    elif config.environment.vec_env_type == "dummy":
        env = DummyVecEnv([make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)])
    else:
        raise ValueError("Unknown vec_env_type")
    env.seed(config.environment.seed)
    env = RLXInfo(env)
    return env
