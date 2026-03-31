from rl_x.environments.custom_mujoco.robocup_soccer.locomotion.mujoco.control_functions.pd import PDControl


def get_control_function(name, env, **kwargs):
    if name == "pd":
        return PDControl(env, **kwargs)
    else:
        raise NotImplementedError
