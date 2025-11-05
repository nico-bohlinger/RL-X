from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.sampling_functions.step_probability import StepProbabilitySampling
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.sampling_functions.step_probability_and_reset import StepProbabilityAndResetSampling
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.sampling_functions.every_step import EveryStepSampling
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.sampling_functions.none import NoneSampling


def get_sampling_function(name, env, **kwargs):
    if name == "step_probability":
        return StepProbabilitySampling(env, **kwargs)
    if name == "step_probability_and_reset":
        return StepProbabilityAndResetSampling(env, **kwargs)
    elif name == "every_step":
        return EveryStepSampling(env, **kwargs)
    elif name == "none":
        return NoneSampling(env, **kwargs)
    else:
        raise NotImplementedError
