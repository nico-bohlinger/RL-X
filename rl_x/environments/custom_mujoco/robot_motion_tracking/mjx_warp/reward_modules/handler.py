from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.reward_modules.default import DefaultReward


def get_reward_module(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    else:
        raise NotImplementedError
