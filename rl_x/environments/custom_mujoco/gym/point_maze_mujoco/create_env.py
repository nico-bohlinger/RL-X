import gymnasium as gym

from rl_x.environments.custom_mujoco.gym.point_maze_mujoco.environment import PointMaze
from rl_x.environments.custom_mujoco.gym.point_maze_mujoco.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.gym.point_maze_mujoco.wrappers import RLXInfo


def create_train_and_eval_env(config):
    make_env = lambda: PointMaze(render=config.environment.render, horizon=config.environment.horizon, reward_style=config.environment.reward_style, flipped=config.environment.flipped, success_radius=config.environment.success_radius)
    train_env = RLXInfo(gym.vector.SyncVectorEnv([make_env for _ in range(config.environment.nr_envs)]))
    train_env.general_properties = GeneralProperties
    train_env.horizon = config.environment.horizon
    train_env.reset(seed=config.environment.seed)
    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    eval_env = RLXInfo(gym.vector.SyncVectorEnv([make_env for _ in range(config.environment.nr_envs)]))
    eval_env.general_properties = GeneralProperties
    eval_env.horizon = config.environment.horizon
    eval_env.reset(seed=config.environment.seed)
    return train_env, eval_env
