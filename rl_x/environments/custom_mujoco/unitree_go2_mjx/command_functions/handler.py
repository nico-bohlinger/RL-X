from rl_x.environments.custom_mujoco.unitree_go2_mjx.command_functions.random import RandomCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    else:
        raise NotImplementedError
