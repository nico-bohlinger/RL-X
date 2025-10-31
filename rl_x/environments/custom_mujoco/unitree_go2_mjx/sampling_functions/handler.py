from rl_x.environments.custom_mujoco.unitree_go2_mjx.sampling_functions.step_probability import StepProbabilitySampling
from rl_x.environments.custom_mujoco.unitree_go2_mjx.sampling_functions.every_step import EveryStepSampling
from rl_x.environments.custom_mujoco.unitree_go2_mjx.sampling_functions.none import NoneSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    elif name == "every_step":
        return EveryStepSampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    else:
        raise NotImplementedError
