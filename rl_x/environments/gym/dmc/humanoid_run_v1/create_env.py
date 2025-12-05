import gymnasium as gym
from dm_control import suite
from shimmy import DmControlCompatibilityV0
from gymnasium.spaces import Dict
from gymnasium.wrappers import FlattenObservation

from rl_x.environments.gym.dmc.humanoid_run_v1.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.gym.dmc.humanoid_run_v1.async_vectorized_wrapper import AsyncVectorEnvWithSkipping
from rl_x.environments.gym.dmc.humanoid_run_v1.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    def make_env(seed):
        def thunk():
            domain_name, task_name = config.environment.type.split("-")
            env = suite.load(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs={"random": config.environment.seed},
            )
            env = DmControlCompatibilityV0(env, render_mode="human" if config.environment.render else None)
            if isinstance(env.observation_space, Dict):
                env = FlattenObservation(env)
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk
    
    make_env_functions = [make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)]

    if config.environment.nr_envs == 1:
        train_env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        train_env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    if config.environment.nr_envs == 1:
        eval_env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        eval_env = AsyncVectorEnvWithSkipping(make_env_functions, config.environment.async_skip_percentage)
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
