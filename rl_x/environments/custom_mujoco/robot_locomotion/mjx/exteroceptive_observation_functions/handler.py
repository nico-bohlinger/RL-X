from rl_x.environments.custom_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.height_over_ground import HeightOverGroundExteroceptiveObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.depth_image import DepthImageExteroceptiveObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.height_samples import HeightSamplesExteroceptiveObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.exteroceptive_observation_functions.none import NoneExteroceptiveObservation


def get_exteroceptive_observation_function(name, env, **kwargs):
    if name == "height_over_ground":
        return HeightOverGroundExteroceptiveObservation(env, **kwargs)
    elif name == "depth_image":
        return DepthImageExteroceptiveObservation(env, **kwargs)
    elif name == "height_samples":
        return HeightSamplesExteroceptiveObservation(env, **kwargs)
    elif name == "none":
        return NoneExteroceptiveObservation(env, **kwargs)
    else:
        raise NotImplementedError
