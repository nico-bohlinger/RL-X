from rl_x.environments.custom_mujoco.ant.mjx.environment import Ant
from rl_x.environments.custom_mujoco.ant.mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = Ant(config.environment)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = Ant(config.environment)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
