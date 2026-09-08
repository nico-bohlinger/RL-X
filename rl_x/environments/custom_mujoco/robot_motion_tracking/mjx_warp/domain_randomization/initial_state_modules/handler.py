from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.initial_state_modules.default import DefaultInitialState


def get_initial_state_module(name, env):
    if name == "default":
        return DefaultInitialState(env)
    else:
        raise NotImplementedError(name)
