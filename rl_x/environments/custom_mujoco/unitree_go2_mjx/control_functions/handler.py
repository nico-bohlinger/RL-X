from rl_x.environments.custom_mujoco.unitree_go2_mjx.control_functions.pd import PDControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    else:
        raise NotImplementedError
