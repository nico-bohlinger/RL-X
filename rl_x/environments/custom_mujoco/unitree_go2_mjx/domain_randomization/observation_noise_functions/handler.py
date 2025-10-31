from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_noise_functions.default import DefaultDRObservationNoise
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_noise_functions.none import NoneDRObservationNoise


def get_observation_noise_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRObservationNoise(env, **kwargs)
    elif name == "none":
        return NoneDRObservationNoise(env, **kwargs)
    else:
        raise NotImplementedError
