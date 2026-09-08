from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.sampling_modules.on_creation import OnCreationSampling
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.sampling_modules.on_reset import OnResetSampling
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.sampling_modules.step_probability import StepProbabilitySampling


def get_sampling_module(name, env, **kwargs):
    if name == "on_creation":
        return OnCreationSampling(env)
    elif name == "on_reset":
        return OnResetSampling(env)
    elif name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    else:
        raise NotImplementedError(name)
