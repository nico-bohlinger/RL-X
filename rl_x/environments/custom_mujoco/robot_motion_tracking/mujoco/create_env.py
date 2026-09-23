import importlib
from pathlib import Path
import gymnasium as gym

from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.environment import MotionTrackingEnv
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.wrappers import RLXInfo, RecordEpisodeStatistics
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    robot_config = importlib.import_module(f"rl_x.environments.custom_mujoco.robot_motion_tracking.robots.{config.environment.train_robot}.robot_config").robot_config.copy()
    robot_config["directory_path"] = Path(__file__).parent.parent / "robots" / config.environment.train_robot

    def make_env(seed, eval_mode=False):
        def thunk():
            env = MotionTrackingEnv(
                robot_config=robot_config,
                runner_mode=config.runner.mode,
                seed=seed,
                render=config.environment.render,
                env_config=config.environment,
                nr_envs=config.environment.nr_envs,
                eval_mode=eval_mode,
            )
            env = RecordEpisodeStatistics(env)
            env.action_space.seed(config.environment.seed)
            env.observation_space.seed(config.environment.seed)
            return env
        return thunk

    make_env_functions = [make_env(config.environment.seed + i) for i in range(config.environment.nr_envs)]

    if config.environment.nr_envs == 1:
        train_env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        train_env = gym.vector.AsyncVectorEnv(make_env_functions, context="spawn")
    train_env = RLXInfo(train_env)
    train_env.horizon = train_env.call("horizon")[0]
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    make_env_functions = [make_env(config.environment.seed + i, eval_mode=True) for i in range(config.environment.nr_envs)]

    if config.environment.nr_envs == 1:
        eval_env = gym.vector.SyncVectorEnv(make_env_functions)
    else:
        eval_env = gym.vector.AsyncVectorEnv(make_env_functions, context="spawn")
    eval_env = RLXInfo(eval_env)
    eval_env.horizon = eval_env.call("horizon")[0]
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
