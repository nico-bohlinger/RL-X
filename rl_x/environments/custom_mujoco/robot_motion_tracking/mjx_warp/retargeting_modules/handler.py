from jax.experimental import enable_x64

from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.retargeting_modules.orpp import OmniRetarget


def get_retargeting_module(name, env):
    with enable_x64():
        if name == "orpp":
            return OmniRetarget(env)
        else:
            raise NotImplementedError(name)
