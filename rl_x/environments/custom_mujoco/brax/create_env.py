from rl_x.environments.custom_mujoco.brax.environment import Env
from rl_x.environments.custom_mujoco.brax.wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv, NormalizeVecObservation, NormalizeVecReward
from rl_x.environments.custom_mujoco.brax.general_properties import GeneralProperties


def create_env(config):
    env, env_params = BraxGymnaxWrapper("hopper"), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    env = NormalizeVecObservation(env)
    env = NormalizeVecReward(env, 0.99) # gamma
    env.close = lambda: None

    env.general_properties = GeneralProperties

    return env
