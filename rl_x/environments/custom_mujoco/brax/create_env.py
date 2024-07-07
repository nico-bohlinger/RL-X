from rl_x.environments.custom_mujoco.brax.environment import Ant
from rl_x.environments.custom_mujoco.brax.general_properties import GeneralProperties


def create_env(config):
    env = Ant()
    env.close = lambda: None

    env.general_properties = GeneralProperties

    return env
