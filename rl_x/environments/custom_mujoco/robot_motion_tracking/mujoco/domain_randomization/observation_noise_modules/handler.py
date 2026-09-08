from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.observation_noise_modules.default import DefaultObservationNoise


def get_observation_noise_module(name, env):
    if name == "default":
        return DefaultObservationNoise(env)
    else:
        raise NotImplementedError(name)
