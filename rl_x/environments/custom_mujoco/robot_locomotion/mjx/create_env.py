import importlib
from pathlib import Path

from rl_x.environments.custom_mujoco.robot_locomotion.mjx.environment import LocomotionEnv
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.general_properties import GeneralProperties


def create_env(config):
    robot_config = importlib.import_module(f"rl_x.environments.custom_mujoco.robot_locomotion.robots.{config.environment.train_robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot

    env = LocomotionEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        env_config=config.environment,
        nr_envs=config.environment.nr_envs,
    )

    env.general_properties = GeneralProperties

    return env
