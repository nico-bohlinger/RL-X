from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_dropout_functions.default import DefaultDRObservationDropout
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.observation_dropout_functions.none import NoneDRObservationDropout



def get_observation_dropout_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRObservationDropout(env, **kwargs)
    elif name == "none":
        return NoneDRObservationDropout(env, **kwargs)
    else:
        raise NotImplementedError
