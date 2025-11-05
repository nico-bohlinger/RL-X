from rl_x.environments.custom_isaac_lab.ant.environment import environment_creator
from rl_x.environments.custom_isaac_lab.ant.general_properties import GeneralProperties


def create_env(config):
    env = environment_creator(config)
    env.general_properties = GeneralProperties

    return env
