from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.mujoco_model_functions.default import DefaultDRMuJoCoModel
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.mujoco_model_functions.none import NoneDRMuJoCoModel


def get_domain_randomization_mujoco_model_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRMuJoCoModel(env, **kwargs)
    elif name == "none":
        return NoneDRMuJoCoModel(env, **kwargs)
    else:
        raise NotImplementedError
