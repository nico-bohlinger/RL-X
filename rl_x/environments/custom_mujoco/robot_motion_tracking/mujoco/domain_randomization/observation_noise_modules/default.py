import numpy as np


class DefaultObservationNoise:
    def __init__(self, env):
        self.env = env

        self.reference_anchor_orientation = env.env_config["domain_randomization"]["observation_noise"]["default"]["reference_anchor_orientation"]
        self.base_angular_velocity = env.env_config["domain_randomization"]["observation_noise"]["default"]["base_angular_velocity"]
        self.joint_position = env.env_config["domain_randomization"]["observation_noise"]["default"]["joint_position"]
        self.joint_velocity = env.env_config["domain_randomization"]["observation_noise"]["default"]["joint_velocity"]
        self.enable_randomization = env.env_config["domain_randomization"]["observation_noise"]["default"]["enable_randomization"]
        scales = np.zeros(env.actor_observation_size, dtype=np.float32)
        scales[env.reference_anchor_orientation_obs_idx] = self.reference_anchor_orientation
        scales[env.base_angular_velocity_obs_idx] = self.base_angular_velocity
        scales[env.joint_positions_obs_idx] = self.joint_position
        scales[env.joint_velocities_obs_idx] = self.joint_velocity
        self.scales = scales


    def apply(self, observation):
        if self.enable_randomization and not self.env.internal_state["in_eval_mode"]:
            observation[self.env.policy_observation_indices] += self.env.np_rng.uniform(-1.0, 1.0, self.scales.shape) * self.scales
        return observation
