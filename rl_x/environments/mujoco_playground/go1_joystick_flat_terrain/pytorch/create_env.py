from mujoco_playground import registry, wrapper, wrapper_torch

from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.pytorch.wrappers import RLXInfo
from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.pytorch.general_properties import GeneralProperties


def create_env(config):
    if config.environment.render:
        raise NotImplementedError("Rendering is not supported yet.")
    
    mbp_env_config = registry.get_default_config(config.environment.type)
    randomizer = (registry.get_domain_randomizer(config.environment.type) if config.environment.use_domain_randomization else None)

    env = registry.load(config.environment.type, config=mbp_env_config)
    env = wrapper_torch.RSLRLBraxWrapper(
        env=env,
        num_actors=config.environment.nr_envs,
        seed=config.environment.seed,
        episode_length=mbp_env_config.episode_length,
        action_repeat=mbp_env_config.action_repeat,
        randomization_fn=randomizer,
        render_callback=None,
        device_rank=0  # TODO: Make this dependent on the selected GPU in a multi-GPU setup
    )

    env = RLXInfo(env, config.environment.nr_envs)
    env.horizon = mbp_env_config.episode_length

    env.general_properties = GeneralProperties

    return env
