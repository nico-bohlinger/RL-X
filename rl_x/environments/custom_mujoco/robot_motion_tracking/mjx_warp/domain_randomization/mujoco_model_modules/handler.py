from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.mujoco_model_modules.default import DefaultMujocoModel


def get_mujoco_model_module(name, env):
    if name == "default":
        return DefaultMujocoModel(env)
    else:
        raise NotImplementedError(name)
