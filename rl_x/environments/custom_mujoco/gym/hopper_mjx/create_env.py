from rl_x.environments.custom_mujoco.gym.hopper_mjx.environment import Hopper
from rl_x.environments.custom_mujoco.gym.hopper_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = Hopper(config.environment.render)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = Hopper(config.environment.render)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
