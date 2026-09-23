from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.termination_modules.tracking_error import TrackingErrorTermination


def get_termination_module(name, env, **kwargs):
    if name == "tracking_error":
        return TrackingErrorTermination(env, **kwargs)
    else:
        raise NotImplementedError
