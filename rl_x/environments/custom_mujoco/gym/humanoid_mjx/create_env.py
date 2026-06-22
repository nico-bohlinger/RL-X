from rl_x.environments.custom_mujoco.gym.humanoid_mjx.environment import Humanoid
from rl_x.environments.custom_mujoco.gym.humanoid_mjx.general_properties import GeneralProperties

def create_train_and_eval_env(config):
    train_env = Humanoid(config.environment.render)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = Humanoid(config.environment.render)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env