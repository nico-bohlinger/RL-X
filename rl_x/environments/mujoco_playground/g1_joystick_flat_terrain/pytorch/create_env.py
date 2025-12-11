from mujoco_playground import registry, wrapper_torch

from rl_x.environments.mujoco_playground.g1_joystick_flat_terrain.pytorch.wrappers import RLXInfo
from rl_x.environments.mujoco_playground.g1_joystick_flat_terrain.pytorch.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    if config.environment.render:
        raise NotImplementedError("Rendering is not supported yet.")
    
    mbp_env_config = registry.get_default_config(config.environment.type)
    randomizer = (registry.get_domain_randomizer(config.environment.type) if config.environment.use_domain_randomization else None)

    train_env = registry.load(config.environment.type, config=mbp_env_config)
    train_env = wrapper_torch.RSLRLBraxWrapper(
        env=train_env,
        num_actors=config.environment.nr_envs,
        seed=config.environment.seed,
        episode_length=mbp_env_config.episode_length,
        action_repeat=mbp_env_config.action_repeat,
        randomization_fn=randomizer,
        render_callback=None,
        device_rank=0  # TODO: Make this dependent on the selected GPU in a multi-GPU setup
    )
    train_env = RLXInfo(train_env, config.environment.nr_envs)
    train_env.horizon = mbp_env_config.episode_length
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = registry.load(config.environment.type, config=mbp_env_config)
    eval_env = wrapper_torch.RSLRLBraxWrapper(
        env=eval_env,
        num_actors=config.environment.nr_envs,
        seed=config.environment.seed,
        episode_length=mbp_env_config.episode_length,
        action_repeat=mbp_env_config.action_repeat,
        randomization_fn=randomizer,
        render_callback=None,
        device_rank=0  # TODO: Make this dependent on the selected GPU in a multi-GPU setup
    )
    eval_env = RLXInfo(eval_env, config.environment.nr_envs)
    eval_env.horizon = mbp_env_config.episode_length
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
