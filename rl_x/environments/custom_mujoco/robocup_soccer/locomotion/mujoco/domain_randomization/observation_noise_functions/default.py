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


    def modify_observation(self, observation):
        observation[self.joint_positions_obs_idx] += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(len(self.joint_positions_obs_idx),), low=-self.joint_position_noise, high=self.joint_position_noise)
        observation[self.imu_angular_vel_obs_idx] += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(len(self.imu_angular_vel_obs_idx),), low=-self.imu_angular_velocity_noise, high=self.imu_angular_velocity_noise)
        observation[self.joint_velocities_obs_idx] += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(len(self.joint_velocities_obs_idx),), low=-self.joint_velocity_noise, high=self.joint_velocity_noise)
        observation[self.gravity_vector_obs_idx] += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(len(self.gravity_vector_obs_idx),), low=-self.gravity_vector_noise, high=self.gravity_vector_noise)
        if len(self.policy_exteroception_obs_idx) > 0:
            observation[self.policy_exteroception_obs_idx] += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(len(self.policy_exteroception_obs_idx),), low=-self.exteroception_noise, high=self.exteroception_noise)
