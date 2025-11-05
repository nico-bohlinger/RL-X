from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.joint_dropout_functions.default import DefaultDRJointDropout
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.joint_dropout_functions.none import NoneDRJointDropout



def get_joint_dropout_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRJointDropout(env, **kwargs)
    elif name == "none":
        return NoneDRJointDropout(env, **kwargs)
    else:
        raise NotImplementedError
