from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.control_modules.pd import PDControl


def get_control_module(name, env):
    if name == "pd":
        return PDControl(env)
    else:
        raise NotImplementedError
