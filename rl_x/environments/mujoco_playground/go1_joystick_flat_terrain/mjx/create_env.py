from mujoco_playground import registry, wrapper

from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.mjx.wrappers import RLXInfo
from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.mjx.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    if config.environment.render:
        raise NotImplementedError("Rendering is not supported yet.")

    mbp_env_config = registry.get_default_config(config.environment.type)
    randomizer = (registry.get_domain_randomizer(config.environment.type) if config.environment.use_domain_randomization else None)

    train_env = registry.load(config.environment.type, config=mbp_env_config)
    train_env = wrapper.wrap_for_brax_training(train_env, episode_length=mbp_env_config.episode_length, action_repeat=mbp_env_config.action_repeat, randomization_fn=randomizer)
    train_env = RLXInfo(train_env, config.environment.nr_envs)
    train_env.horizon = mbp_env_config.episode_length
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = registry.load(config.environment.type, config=mbp_env_config)
    eval_env = wrapper.wrap_for_brax_training(eval_env, episode_length=mbp_env_config.episode_length, action_repeat=mbp_env_config.action_repeat, randomization_fn=randomizer)
    eval_env = RLXInfo(eval_env, config.environment.nr_envs)
    eval_env.horizon = mbp_env_config.episode_length
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
