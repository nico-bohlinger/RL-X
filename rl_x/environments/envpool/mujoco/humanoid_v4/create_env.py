import envpool

from rl_x.environments.envpool.mujoco.humanoid_v4.wrappers import RLXInfo
from rl_x.environments.envpool.mujoco.humanoid_v4.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = envpool.make(config.environment.type, env_type="gymnasium", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties
    train_env.action_space = train_env.action_space
    train_env.observation_space = train_env.observation_space
    train_env.single_action_space = train_env.action_space
    train_env.single_observation_space = train_env.observation_space
    train_env.action_space.seed(config.environment.seed)
    train_env.observation_space.seed(config.environment.seed)

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = envpool.make(config.environment.type, env_type="gymnasium", seed=config.environment.seed, num_envs=config.environment.nr_envs)
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties
    eval_env.action_space = eval_env.action_space
    eval_env.observation_space = eval_env.observation_space
    eval_env.single_action_space = eval_env.action_space
    eval_env.single_observation_space = eval_env.observation_space
    eval_env.action_space.seed(config.environment.seed)
    eval_env.observation_space.seed(config.environment.seed)

    return train_env, eval_env
