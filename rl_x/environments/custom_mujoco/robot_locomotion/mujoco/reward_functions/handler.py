from rl_x.environments.custom_mujoco.robot_locomotion.mujoco.reward_functions.default import DefaultReward


def get_reward_function(name, env, **kwargs):
    if name == "default":
        return DefaultReward(env, **kwargs)
    else:
        raise NotImplementedError
