from rl_x.environments.custom_mujoco.brax.environment import Ant
from rl_x.environments.custom_mujoco.brax.wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, NormalizeVecObservation, NormalizeVecReward
from rl_x.environments.custom_mujoco.brax.general_properties import GeneralProperties


def create_env(config):
    env = Ant()
    # env = BraxGymnaxWrapper(env)
    # env = LogWrapper(env)
    # env = ClipAction(env)
    # env = NormalizeVecObservation(env)
    # env = NormalizeVecReward(env, 0.99)
    env.close = lambda: None

    env.general_properties = GeneralProperties

    return env
