import gymnasium as gym
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.spaces import Dict
from gymnasium.wrappers import FlattenObservation

from rl_x.environments.gym.dmc.humanoid_run_v1.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.gym.dmc.humanoid_run_v1.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.gym.dmc.humanoid_run_v1.general_properties import GeneralProperties


def create_env(config):
    def make_env(seed):
        def thunk():
            domain_name, task_name = config.environment.type.split("-")
            env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs={"random": config.environment.seed},
            )
            env = DmControlCompatibilityV0(env, render_mode="rgb_array")
            if isinstance(env.observation_space, Dict):
                env = FlattenObservation(env)
            env = RecordEpisodeStatistics(env)
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

    return env
