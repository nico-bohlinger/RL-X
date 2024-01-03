from rl_x.environments.custom_mujoco.ant_mjx.environment import Ant
from rl_x.environments.custom_mujoco.ant_mjx.wrappers import RLXInfo, GymWrapper
from rl_x.environments.custom_mujoco.ant_mjx.general_properties import GeneralProperties


def create_env(config):
    env = Ant()
    env = GymWrapper(env, seed=config.environment.seed, nr_envs=config.environment.nr_envs)
    env = RLXInfo(env)
    env.general_properties = GeneralProperties

    return env
