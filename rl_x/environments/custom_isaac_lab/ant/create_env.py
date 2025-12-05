from rl_x.environments.custom_isaac_lab.ant.environment import environment_creator
from rl_x.environments.custom_isaac_lab.ant.general_properties import GeneralProperties


def create_train_and_eval_env(config):
    train_env = environment_creator(config)
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env
    
    eval_env = environment_creator(config)
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
