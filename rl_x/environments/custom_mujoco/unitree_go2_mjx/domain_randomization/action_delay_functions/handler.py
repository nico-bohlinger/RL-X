from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.action_delay_functions.default import DefaultActionDelay
from rl_x.environments.custom_mujoco.unitree_go2_mjx.domain_randomization.action_delay_functions.none import NoneActionDelay


def get_get_domain_randomization_action_delay_function(name, env, **kwargs):
    if name == "none":
        return NoneActionDelay(env, **kwargs)
    elif name == "default":
        return DefaultActionDelay(env, **kwargs)
    else:
        raise NotImplementedError
