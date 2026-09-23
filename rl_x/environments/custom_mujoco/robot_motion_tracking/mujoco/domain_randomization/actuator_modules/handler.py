from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.actuator_modules.default import DefaultActuator


def get_actuator_module(name, env):
    if name == "default":
        return DefaultActuator(env)
    else:
        raise NotImplementedError(name)
