from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.control_functions.default import DefaultDRControl
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.control_functions.none import NoneDRControl


def get_domain_randomization_control_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRControl(env, **kwargs)
    elif name == "none":
        return NoneDRControl(env, **kwargs)
    else:
        raise NotImplementedError
