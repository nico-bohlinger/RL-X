import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.unitree_go2_mjx import observation_indices as obs_idx


class DefaultDRObservationNoise:
    def __init__(self, env):
        self.env = env

        self.joint_position_noise = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["joint_position"]
        self.joint_velocity_noise = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["joint_velocity"]
        self.trunk_angular_velocity_noise = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["trunk_angular_velocity"]
        self.ground_contact_noise_chance = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["ground_contact_noise_chance"]
        self.contact_time_noise_chance = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["contact_time_noise_chance"]
        self.contact_time_noise_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["contact_time_noise_factor"]
        self.gravity_vector_noise = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["gravity_vector"]
        self.exteroception_noise = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_noise"]["exteroception"]

        self.joint_position_ids = jnp.array([joint_indices[0] for joint_indices in self.env.joint_observation_indices])
        self.joint_velocity_ids = jnp.array([joint_indices[1] for joint_indices in self.env.joint_observation_indices])
        self.foot_contact_ids = jnp.array([foot_indices[0] for foot_indices in self.env.feet_observation_indices])
        self.foot_time_since_last_touchdown_ids = jnp.array([foot_indices[1] for foot_indices in self.env.feet_observation_indices])
        self.trunk_angular_velocity_ids = obs_idx.TRUNK_ANGULAR_VELOCITIES
        self.gravity_vector_ids = obs_idx.PROJECTED_GRAVITY
        self.exteroception_ids = obs_idx.EXTEROCEPTION


    def modify_observation(self, observation, key):
        noise_key1, noise_key2, noise_key3, noise_key4, noise_key5, noise_key6, noise_key7 = jax.random.split(key, 7)

        observation = observation.at[self.joint_position_ids].set(observation[self.joint_position_ids] + jax.random.uniform(noise_key1, shape=(len(self.joint_position_ids),), minval=-self.joint_position_noise, maxval=self.joint_position_noise))
        observation = observation.at[self.trunk_angular_velocity_ids].set(observation[self.trunk_angular_velocity_ids] + jax.random.uniform(noise_key2, shape=(len(self.trunk_angular_velocity_ids),), minval=-self.trunk_angular_velocity_noise, maxval=self.trunk_angular_velocity_noise))
        observation = observation.at[self.joint_velocity_ids].set(observation[self.joint_velocity_ids] + jax.random.uniform(noise_key3, shape=(len(self.joint_velocity_ids),), minval=-self.joint_velocity_noise, maxval=self.joint_velocity_noise))
        observation = observation.at[self.gravity_vector_ids].set(observation[self.gravity_vector_ids] + jax.random.uniform(noise_key4, shape=(len(self.gravity_vector_ids),), minval=-self.gravity_vector_noise, maxval=self.gravity_vector_noise))
        observation = observation.at[self.foot_contact_ids].set(jnp.where(jax.random.uniform(noise_key5, shape=(len(self.foot_contact_ids),)) < self.ground_contact_noise_chance, 1 - observation[self.foot_contact_ids], observation[self.foot_contact_ids]))
        observation = observation.at[self.foot_time_since_last_touchdown_ids].set(jnp.where(jax.random.uniform(noise_key6, shape=(len(self.foot_time_since_last_touchdown_ids),)) < self.contact_time_noise_chance, observation[self.foot_time_since_last_touchdown_ids] + jax.random.uniform(noise_key6, shape=(len(self.foot_time_since_last_touchdown_ids),), minval=-self.contact_time_noise_factor * self.env.dt, maxval=self.contact_time_noise_factor * self.env.dt), observation[self.foot_time_since_last_touchdown_ids]))
        observation = observation.at[self.exteroception_ids].set(observation[self.exteroception_ids] + jax.random.uniform(noise_key7, shape=(len(self.exteroception_ids),), minval=-self.exteroception_noise, maxval=self.exteroception_noise))

        return observation
