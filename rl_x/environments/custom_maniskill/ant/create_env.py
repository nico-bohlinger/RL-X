import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import rl_x.environments.custom_maniskill.ant.environment
from rl_x.environments.custom_maniskill.ant.wrappers import RLXInfo, RenderWrapper
from rl_x.environments.custom_maniskill.ant.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    if config.environment.device == "cpu" and config.environment.nr_envs > 1:
        raise NotImplementedError("When using the CPU, parallel environments are not supported yet.")

    train_env = gym.make(
        "ManiSkill-Ant",
        num_envs=config.environment.nr_envs,
        sim_backend="physx_cuda" if config.environment.device == "gpu" else "physx_cpu",
        render_mode="human" if config.environment.render else None,
        parallel_in_single_scene=True if config.environment.render and config.environment.nr_envs > 1 else False,
    )
    train_env = ManiSkillVectorEnv(train_env, auto_reset=True, ignore_terminations=False, record_metrics=True)
    if config.environment.render:
        train_env = RenderWrapper(train_env)
    train_env = RLXInfo(train_env)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = gym.make(
        "ManiSkill-Ant",
        num_envs=config.environment.nr_envs,
        sim_backend="physx_cuda" if config.environment.device == "gpu" else "physx_cpu",
        render_mode="human" if config.environment.render else None,
        parallel_in_single_scene=True if config.environment.render and config.environment.nr_envs > 1 else False,
    )
    eval_env = ManiSkillVectorEnv(eval_env, auto_reset=True, ignore_terminations=False, record_metrics=True)
    if config.environment.render:
        eval_env = RenderWrapper(eval_env)
    eval_env = RLXInfo(eval_env)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
