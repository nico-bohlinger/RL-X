import jax


class DefaultDRObservationNoise:
    def __init__(self, env):
        self.env = env

        self.joint_position_noise = env.env_config["domain_randomization"]["observation_noise"]["joint_position"]
        self.joint_velocity_noise = env.env_config["domain_randomization"]["observation_noise"]["joint_velocity"]
        self.imu_angular_velocity_noise = env.env_config["domain_randomization"]["observation_noise"]["imu_angular_velocity"]
        self.gravity_vector_noise = env.env_config["domain_randomization"]["observation_noise"]["gravity_vector"]
        self.exteroception_noise = env.env_config["domain_randomization"]["observation_noise"]["exteroception"]


    def init_attributes(self):
        self.joint_positions_obs_idx = self.env.joint_positions_obs_idx
        self.joint_velocities_obs_idx = self.env.joint_velocities_obs_idx
        self.imu_angular_vel_obs_idx = self.env.imu_angular_vel_obs_idx
        self.gravity_vector_obs_idx = self.env.gravity_vector_obs_idx
        self.policy_exteroception_obs_idx = self.env.policy_exteroception_obs_idx


    def modify_observation(self, internal_state, observation, key):
        noise_key1, noise_key2, noise_key3, noise_key4, noise_key5 = jax.random.split(key, 5)

        observation = observation.at[self.joint_positions_obs_idx].set(observation[self.joint_positions_obs_idx] + internal_state["env_curriculum_coeff"] * jax.random.uniform(noise_key1, shape=(len(self.joint_positions_obs_idx),), minval=-self.joint_position_noise, maxval=self.joint_position_noise))
        observation = observation.at[self.imu_angular_vel_obs_idx].set(observation[self.imu_angular_vel_obs_idx] + internal_state["env_curriculum_coeff"] * jax.random.uniform(noise_key2, shape=(len(self.imu_angular_vel_obs_idx),), minval=-self.imu_angular_velocity_noise, maxval=self.imu_angular_velocity_noise))
        observation = observation.at[self.joint_velocities_obs_idx].set(observation[self.joint_velocities_obs_idx] + internal_state["env_curriculum_coeff"] * jax.random.uniform(noise_key3, shape=(len(self.joint_velocities_obs_idx),), minval=-self.joint_velocity_noise, maxval=self.joint_velocity_noise))
        observation = observation.at[self.gravity_vector_obs_idx].set(observation[self.gravity_vector_obs_idx] + internal_state["env_curriculum_coeff"] * jax.random.uniform(noise_key4, shape=(len(self.gravity_vector_obs_idx),), minval=-self.gravity_vector_noise, maxval=self.gravity_vector_noise))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(observation[self.policy_exteroception_obs_idx] + internal_state["env_curriculum_coeff"] * jax.random.uniform(noise_key5, shape=(len(self.policy_exteroception_obs_idx),), minval=-self.exteroception_noise, maxval=self.exteroception_noise))

        return observation
