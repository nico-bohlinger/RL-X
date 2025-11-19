from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.unseen_robot_functions.default import DefaultDRUnseenRobotFunction
from rl_x.environments.custom_mujoco.robot_locomotion.mjx.domain_randomization.unseen_robot_functions.none import NoneDRUnseenRobotFunction


def get_domain_randomization_unseen_robot_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRUnseenRobotFunction(env, **kwargs)
    elif name == "none":
        return NoneDRUnseenRobotFunction(env, **kwargs)
    else:
        raise NotImplementedError
