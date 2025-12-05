import numpy as np
import envpool
from gymnasium.spaces import Box

from rl_x.environments.envpool.dmc.humanoid_run_v1.wrappers import RLXInfo
from rl_x.environments.envpool.dmc.humanoid_run_v1.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = envpool.make("HumanoidRun-v1", env_type="dm", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    as_spec = train_env.action_spec()
    train_env.action_space = Box(low=np.full(as_spec.shape, as_spec.minimum), high=np.full(as_spec.shape, as_spec.maximum), shape=as_spec.shape, dtype=np.float32, seed=config.environment.seed)
    os_shape = (95,)
    train_env.observation_space = Box(low=np.full(os_shape, 0.0), high=np.full(os_shape, np.inf), shape=os_shape, dtype=np.float32, seed=config.environment.seed)
    train_env.single_action_space = train_env.action_space
    train_env.single_observation_space = train_env.observation_space
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = envpool.make("HumanoidRun-v1", env_type="dm", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    as_spec = eval_env.action_spec()
    eval_env.action_space = Box(low=np.full(as_spec.shape, as_spec.minimum), high=np.full(as_spec.shape, as_spec.maximum), shape=as_spec.shape, dtype=np.float32, seed=config.environment.seed)
    os_shape = (95,)
    eval_env.observation_space = Box(low=np.full(os_shape, 0.0), high=np.full(os_shape, np.inf), shape=os_shape, dtype=np.float32, seed=config.environment.seed)
    eval_env.single_action_space = eval_env.action_space
    eval_env.single_observation_space = eval_env.observation_space
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
