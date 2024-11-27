from rl_x.environments.custom_mujoco.unitree_go2_mjx.termination_functions.below_ground_and_power import BelowGroundAndPowerTermination


def get_termination_function(name, env, **kwargs):
    if name == "below_ground_and_power":
        return BelowGroundAndPowerTermination(env, **kwargs)
    else:
        raise NotImplementedError
