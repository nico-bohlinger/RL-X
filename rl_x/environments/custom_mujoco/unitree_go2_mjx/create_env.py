import importlib
from pathlib import Path

from rl_x.environments.custom_mujoco.unitree_go2_mjx.environment import GeneralLocomotionEnv
from rl_x.environments.custom_mujoco.unitree_go2_mjx.general_properties import GeneralProperties


def create_env(config):
    robot_config = importlib.import_module(f"loco_mjx.environments.robots.{config.environment.robot}.robot_config").robot_config
    robot_config["directory_path"] = Path(__file__).parent.parent.parent.parent / "robots" / config.environment.robot

    locomotion_env_name = Path(__file__).parent.parent.name
    if locomotion_env_name not in robot_config["locomotion_envs"]:
        raise ValueError(f"Robot '{config.environment.robot}' is not compatible with locomotion environment '{locomotion_env_name}'. Only the following locomotion environments are supported: {robot_config['locomotion_envs']}")

    env = GeneralLocomotionEnv(
        robot_config=robot_config,
        runner_mode=config.runner.mode,
        render=config.environment.render,
        control_type=config.environment.control_type,
        command_type=config.environment.command_type,
        command_sampling_type=config.environment.command_sampling_type,
        initial_state_type=config.environment.initial_state_type,
        reward_type=config.environment.reward_type,
        termination_type=config.environment.termination_type,
        domain_randomization_sampling_type=config.environment.domain_randomization_sampling_type,
        domain_randomization_action_delay_type=config.environment.domain_randomization_action_delay_type,
        domain_randomization_mujoco_model_type=config.environment.domain_randomization_mujoco_model_type,
        domain_randomization_control_type=config.environment.domain_randomization_control_type,
        domain_randomization_perturbation_sampling_type=config.environment.domain_randomization_perturbation_sampling_type,
        domain_randomization_perturbation_type=config.environment.domain_randomization_perturbation_type,
        observation_noise_type=config.environment.observation_noise_type,
        observation_dropout_type=config.environment.observation_dropout_type,
        exteroceptive_observation_type=config.environment.exteroceptive_observation_type,
        terrain_type=config.environment.terrain_type,
        add_goal_arrow=config.environment.add_goal_arrow,
        timestep=config.environment.timestep,
        episode_length_in_seconds=config.environment.episode_length_in_seconds,
        total_nr_envs=config.environment.nr_envs,
    )

    env.general_properties = GeneralProperties

    return env
