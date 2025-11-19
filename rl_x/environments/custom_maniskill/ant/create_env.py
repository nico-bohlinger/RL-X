import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import rl_x.environments.custom_maniskill.ant.environment
from rl_x.environments.custom_maniskill.ant.wrappers import RLXInfo, RenderWrapper
from rl_x.environments.custom_maniskill.ant.general_properties import GeneralProperties


def create_env(config):
    if config.environment.device == "cpu" and config.environment.nr_envs > 1:
        raise NotImplementedError("When using the CPU, parallel environments are not supported yet.")

    env = gym.make(
        "ManiSkill-Ant",
        num_envs=config.environment.nr_envs,
        sim_backend="physx_cuda" if config.environment.device == "gpu" else "physx_cpu",
        render_mode="human" if config.environment.render else None,
        parallel_in_single_scene=True if config.environment.render and config.environment.nr_envs > 1 else False,
    )
    env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False, record_metrics=True)

    if config.environment.render:
        env = RenderWrapper(env)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    return env
