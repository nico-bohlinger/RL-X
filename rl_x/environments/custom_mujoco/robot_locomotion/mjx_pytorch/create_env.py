import importlib
from pathlib import Path

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_pytorch.wrappers import RLXInfo
from rl_x.environments.custom_mujoco.robot_locomotion.mjx_pytorch.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    robot_config = importlib.import_module(f"rl_x.environments.custom_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot

    train_env = LocomotionEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )
    train_env = RLXInfo(train_env, config.environment.nr_envs, config.environment.seed)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = LocomotionEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )
    eval_env = RLXInfo(eval_env, config.environment.nr_envs, config.environment.seed)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
