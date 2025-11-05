from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.exteroceptive_observation_functions.height_over_ground import HeightOverGroundExteroceptiveObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.exteroceptive_observation_functions.height_samples import HeightSamplesExteroceptiveObservation
from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.exteroceptive_observation_functions.none import NoneExteroceptiveObservation


def get_exteroceptive_observation_function(name, env, **kwargs):
    if name == "height_over_ground":
        return HeightOverGroundExteroceptiveObservation(env, **kwargs)
    elif name == "depth_image":
        raise NotImplementedError("Depth image is not ported from MJX yet")
    elif name == "height_samples":
        return HeightSamplesExteroceptiveObservation(env, **kwargs)
    elif name == "none":
        return NoneExteroceptiveObservation(env, **kwargs)
    else:
        raise NotImplementedError
