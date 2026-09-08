from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.action_delay_modules.default import DefaultActionDelay


def get_action_delay_module(name, env):
    if name == "default":
        return DefaultActionDelay(env)
    else:
        raise NotImplementedError(name)
