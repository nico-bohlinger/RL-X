import numpy as np
import jax
import jax.numpy as jnp


class DefaultObservationNoise:
    def __init__(self, env):
        self.env = env

        self.reference_anchor_orientation = env.env_config["domain_randomization"]["observation_noise"]["default"]["reference_anchor_orientation"]
        self.base_angular_velocity = env.env_config["domain_randomization"]["observation_noise"]["default"]["base_angular_velocity"]
        self.joint_position = env.env_config["domain_randomization"]["observation_noise"]["default"]["joint_position"]
        self.joint_velocity = env.env_config["domain_randomization"]["observation_noise"]["default"]["joint_velocity"]
        self.enable_randomization = env.env_config["domain_randomization"]["observation_noise"]["default"]["enable_randomization"]
        scales = np.zeros(env.actor_observation_size, dtype=np.float32)
        scales[np.asarray(env.reference_anchor_orientation_obs_idx)] = self.reference_anchor_orientation
        scales[np.asarray(env.base_angular_velocity_obs_idx)] = self.base_angular_velocity
        scales[np.asarray(env.joint_positions_obs_idx)] = self.joint_position
        scales[np.asarray(env.joint_velocities_obs_idx)] = self.joint_velocity
        self.scales = jnp.asarray(scales)


    def apply(self, key, observation, eval_mode):
        noise = jax.random.uniform(key, observation[..., self.env.policy_observation_indices].shape, minval=-1.0, maxval=1.0) * self.scales

        return observation.at[..., self.env.policy_observation_indices].add(jnp.where(self.enable_randomization & ~eval_mode, noise, 0.0))
