from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.perturbation_modules.default import DefaultPerturbation


def get_perturbation_module(name, env):
    if name == "default":
        return DefaultPerturbation(env)
    else:
        raise NotImplementedError(name)
