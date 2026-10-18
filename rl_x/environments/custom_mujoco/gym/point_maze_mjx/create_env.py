from rl_x.environments.custom_mujoco.gym.point_maze_mjx.environment import PointMaze
from rl_x.environments.custom_mujoco.gym.point_maze_mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = PointMaze(render=config.environment.render)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = PointMaze(render=config.environment.render)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
