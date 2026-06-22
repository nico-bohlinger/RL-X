from rl_x.environments.custom_mujoco.gym.walker2d_mjx.environment import Walker2D
from rl_x.environments.custom_mujoco.gym.walker2d_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = Walker2D(config.environment.render)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = Walker2D(config.environment.render)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
