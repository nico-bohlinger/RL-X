from rl_x.environments.custom_mujoco.ant.mjx.environment import Ant
from rl_x.environments.custom_mujoco.ant.mjx.general_properties import GeneralProperties


def create_env(config):
    env = Ant(config.environment.render)

    env.general_properties = GeneralProperties

    return env
