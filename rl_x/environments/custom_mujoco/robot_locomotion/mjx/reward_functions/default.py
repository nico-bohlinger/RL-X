import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env):
        self.env = env

        self.tracking_xy_velocity_command_coeff = env.env_config["reward"]["tracking_xy_velocity_command_coeff"] * env.dt
        self.tracking_xy_temperature = env.env_config["reward"]["tracking_xy_temperature"]
        self.tracking_yaw_velocity_command_coeff = env.env_config["reward"]["tracking_yaw_velocity_command_coeff"] * env.dt
        self.tracking_yaw_temperature = env.env_config["reward"]["tracking_yaw_temperature"]
        self.alive_clipped_coeff = env.env_config["reward"]["alive_clipped_coeff"] * env.dt
        self.alive_unclipped_coeff = env.env_config["reward"]["alive_unclipped_coeff"] * env.dt
        self.z_velocity_coeff = env.env_config["reward"]["z_velocity_coeff"] * env.dt
        self.imu_acceleration_coeff = env.env_config["reward"]["imu_acceleration_coeff"] * env.dt
        self.roll_pitch_vel_coeff = env.env_config["reward"]["roll_pitch_vel_coeff"] * env.dt
        self.roll_pitch_pos_coeff = env.env_config["reward"]["roll_pitch_pos_coeff"] * env.dt
        self.actuator_joint_nominal_diff_coeff = env.env_config["reward"]["actuator_joint_nominal_diff_coeff"] * env.dt
        self.joint_position_limit_coeff = env.env_config["reward"]["joint_position_limit_coeff"] * env.dt
        self.soft_joint_position_limit = env.env_config["reward"]["soft_joint_position_limit"]
        self.actuator_joint_velocity_limit_coeff = env.env_config["reward"]["actuator_joint_velocity_limit_coeff"] * env.dt
        self.soft_actuator_joint_velocity_limit = env.env_config["reward"]["soft_actuator_joint_velocity_limit"]
        self.joint_velocity_coeff = env.env_config["reward"]["joint_velocity_coeff"] * env.dt
        self.joint_acceleration_coeff = env.env_config["reward"]["joint_acceleration_coeff"] * env.dt
        self.joint_torque_coeff = env.env_config["reward"]["joint_torque_coeff"] * env.dt
        self.power_draw_penalty_coeff = env.env_config["reward"]["power_draw_penalty_coeff"] * env.dt
        self.action_rate_coeff = env.env_config["reward"]["action_rate_coeff"] * env.dt
        self.action_smoothness_coeff = env.env_config["reward"]["action_smoothness_coeff"] * env.dt
        self.collision_coeff = env.env_config["reward"]["collision_coeff"] * env.dt
        self.base_height_coeff = env.env_config["reward"]["base_height_coeff"] * env.dt
        self.foot_air_time_coeff = env.env_config["reward"]["foot_air_time_coeff"] * env.dt
        self.foot_air_time_per_robot_size_m = env.env_config["reward"]["foot_air_time_per_robot_size_m"]
        self.symmetry_air_coeff = env.env_config["reward"]["symmetry_air_coeff"] * env.dt
        self.foot_slip_coeff = env.env_config["reward"]["foot_slip_coeff"] * env.dt
        self.foot_z_velocity_coeff = env.env_config["reward"]["foot_z_velocity_coeff"] * env.dt
        self.foot_flat_contact_coeff = env.env_config["reward"]["foot_flat_contact_coeff"] * env.dt

        self.feet_symmetry_pairs = env.feet_symmetry_pairs


    def init(self, internal_state, mjx_model):
        internal_state["joint_position_limits"] = self.calculate_joint_position_limits(mjx_model)
        self.setup(internal_state)


    def calculate_joint_position_limits(self, mjx_model):
        joint_limits = mjx_model.jnt_range[1:]
        joint_limits_midpoint = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        joint_limits_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower_joint_limits = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        upper_joint_limits = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit
        return jnp.stack([lower_joint_limits, upper_joint_limits], axis=1)

    
    def handle_model_change(self, internal_state, mjx_model, should_change):
        internal_state["joint_position_limits"] = jnp.where(should_change, self.calculate_joint_position_limits(mjx_model), internal_state["joint_position_limits"])


    def setup(self, internal_state):
        internal_state["feet_time_on_ground"] = jnp.zeros(self.env.nr_feet)
        internal_state["feet_time_in_air"] = jnp.zeros(self.env.nr_feet)
        internal_state["previous_actuator_joint_velocities"] = jnp.zeros(self.env.nr_actuator_joints)
        internal_state["previous_imu_linear_velocity"] = jnp.zeros(self.env.imu_linear_velocity_sensor_dim)
        internal_state["sum_tracking_performance_percentage"] = 0.0


    def step(self, data, internal_state):
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        internal_state["feet_time_on_ground"] = jnp.where(feet_floor_contacts, internal_state["feet_time_on_ground"] + self.env.dt, 0.0)
        internal_state["feet_time_in_air"] = jnp.where(feet_floor_contacts, 0.0, internal_state["feet_time_in_air"] + self.env.dt)
        internal_state["previous_actuator_joint_velocities"] = data.qvel[self.env.actuator_joint_mask_qvel]
        internal_state["previous_imu_linear_velocity"] = data.sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        curriculum_coeff = internal_state["env_curriculum_coeff"]
        
        # Tracking velocity command reward
        current_imu_linear_velocity = data.sensordata[self.env.imu_linear_velocity_sensor_adr:self.env.imu_linear_velocity_sensor_adr + self.env.imu_linear_velocity_sensor_dim]
        desired_imu_linear_velocity_xy = internal_state["goal_velocities"][:2]
        xy_difference = desired_imu_linear_velocity_xy - current_imu_linear_velocity[:2]
        xy_velocity_difference_norm = jnp.sum(jnp.square(xy_difference))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * jnp.exp(-xy_velocity_difference_norm / self.tracking_xy_temperature)

        # Tracking angular velocity command reward
        current_imu_angular_velocity = data.sensordata[self.env.imu_angular_velocity_sensor_adr:self.env.imu_angular_velocity_sensor_adr + self.env.imu_angular_velocity_sensor_dim]
        desired_imu_yaw_velocity = internal_state["goal_velocities"][2]
        yaw_velocity_difference_norm = jnp.square(current_imu_angular_velocity[2] - desired_imu_yaw_velocity)
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * jnp.exp(-yaw_velocity_difference_norm / self.tracking_yaw_temperature)

        # Alive clipped reward
        alive_clipped_reward = curriculum_coeff * self.alive_clipped_coeff * 1.0

        # Alive unclipped reward
        alive_unclipped_reward = curriculum_coeff * self.alive_unclipped_coeff * 1.0

        # Z velocity reward
        z_velocity_squared = current_imu_linear_velocity[2] ** 2
        z_velocity_reward = curriculum_coeff * self.z_velocity_coeff * -z_velocity_squared

        # IMU acceleration reward
        imu_acceleration_norm = jnp.mean(jnp.square((current_imu_linear_velocity - internal_state["previous_imu_linear_velocity"]) / self.env.dt))
        imu_acceleration_reward = curriculum_coeff * self.imu_acceleration_coeff * -imu_acceleration_norm

        # Angular velocity reward
        angular_velocity_norm = jnp.sum(jnp.square(current_imu_angular_velocity[:2]))
        angular_velocity_reward = curriculum_coeff * self.roll_pitch_vel_coeff * -angular_velocity_norm

        # Angular position reward
        roll_pitch_position_norm = jnp.sum(jnp.square(internal_state["imu_orientation_euler"][:2]))
        angular_position_reward = curriculum_coeff * self.roll_pitch_pos_coeff * -roll_pitch_position_norm

        # Joint nominal position difference reward
        actuator_joint_nominal_diff_norm = jnp.mean(jnp.square((data.qpos[self.env.actuator_joint_mask_qpos] * internal_state["actuator_joint_keep_nominal"]) - (internal_state["actuator_joint_nominal_positions"] * internal_state["actuator_joint_keep_nominal"])))
        actuator_joint_nominal_diff_reward = curriculum_coeff * self.actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

        # Joint position limit reward
        joint_positions = data.qpos[self.env.actuator_joint_mask_qpos]
        lower_limit_penalty = -jnp.minimum(joint_positions - internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 0], 0.0).mean()
        upper_limit_penalty = jnp.maximum(joint_positions - internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 1], 0.0).mean()
        joint_position_limit_reward = curriculum_coeff * self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Actuator joint velocity limit reward
        actuator_joint_abs_velocities = jnp.abs(data.qvel[self.env.actuator_joint_mask_qvel])
        soft_actuator_joint_velocity_limit = self.soft_actuator_joint_velocity_limit * internal_state["actuator_joint_max_velocities"]
        velocity_limit_penalty = jnp.maximum(actuator_joint_abs_velocities - soft_actuator_joint_velocity_limit, 0.0).mean()
        joint_velocity_limit_reward = curriculum_coeff * self.actuator_joint_velocity_limit_coeff * -velocity_limit_penalty

        # Joint velocity reward
        joint_velocity_norm = jnp.mean(jnp.square(data.qvel[self.env.actuator_joint_mask_qvel]))
        joint_velocity_reward = curriculum_coeff * self.joint_velocity_coeff * -joint_velocity_norm

        # Joint acceleration reward
        acceleration_norm = jnp.mean(jnp.square((internal_state["previous_actuator_joint_velocities"] - data.qvel[self.env.actuator_joint_mask_qvel]) / self.env.dt))
        acceleration_reward = curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = jnp.mean(jnp.square(data.qfrc_actuator[self.env.actuator_joint_mask_qvel]))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # Power draw penalty reward
        power_draw = jnp.mean(jnp.maximum(data.qfrc_actuator[self.env.actuator_joint_mask_qvel] * data.qvel[self.env.actuator_joint_mask_qvel], 0.0))
        power_draw_penalty_reward = curriculum_coeff * self.power_draw_penalty_coeff * -power_draw

        # Action rate reward
        action_rate_norm = jnp.mean(jnp.square(action - internal_state["last_action"]))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm
        
        # Action smoothness reward
        action_smoothness_norm = jnp.mean(jnp.square(action - 2 * internal_state["last_action"] + internal_state["second_last_action"]))
        action_smoothness_reward = curriculum_coeff * self.action_smoothness_coeff * -action_smoothness_norm

        # Collision reward
        all_contact_relevant_geom_xpos = data.geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = mjx_model.geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = jnp.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (jnp.sum(contact_between_geoms) - self.env.reward_collision_sphere_geom_ids.shape[0]) // 2
        nr_collisions = jnp.maximum(nr_collisions - internal_state["nr_collisions_in_nominal"], 0)
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions

        # Walking height
        height_difference_squared = (internal_state["robot_imu_height_over_ground"] - internal_state["robot_nominal_imu_height_over_ground"]) ** 2
        base_height_reward = curriculum_coeff * self.base_height_coeff * -height_difference_squared

        # Foot air time reward
        feet_floor_contacts = self.env.terrain_function.check_feet_floor_contact(data)
        is_standing_command = jnp.all(internal_state["goal_velocities"] == 0.0)
        target_foot_air_time = self.foot_air_time_per_robot_size_m * internal_state["robot_dimensions_mean"]
        target_foot_air_time = (~is_standing_command) * target_foot_air_time
        air_time_reward = jnp.mean(feet_floor_contacts * jnp.minimum(internal_state["feet_time_in_air"] - target_foot_air_time, 0.0))
        foot_air_time_reward = curriculum_coeff * self.foot_air_time_coeff * air_time_reward

        # Symmetry reward
        symmetry_air_violations = jnp.mean(jnp.where((~feet_floor_contacts[self.feet_symmetry_pairs[:, 0]]) & (~feet_floor_contacts[self.feet_symmetry_pairs[:, 1]]), 1, 0))
        symmetry_air_reward = curriculum_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        # Foot slip reward
        feet_global_linear_velocity_x = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start]
        feet_global_linear_velocity_y = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 1]
        feet_global_linear_velocity_xy_norm = jnp.square(feet_global_linear_velocity_x) + jnp.square(feet_global_linear_velocity_y)
        contact_filtered_feet_slip = jnp.mean(feet_floor_contacts * feet_global_linear_velocity_xy_norm)
        foot_slip_reward = curriculum_coeff * self.foot_slip_coeff * -contact_filtered_feet_slip

        # Foot z velocity reward
        feet_global_linear_velocity_z = data.sensordata[self.env.feet_global_linear_velocity_sensor_adrs_start + 2]
        squared_negative_z_velocity = jnp.mean(jnp.square(jnp.minimum(feet_global_linear_velocity_z, 0.0)))
        foot_z_velocity_reward = curriculum_coeff * self.foot_z_velocity_coeff * -squared_negative_z_velocity

        # Foot flat contact reward
        missing_lower_feet_contacts = self.env.terrain_function.check_flat_feet_floor_missing_contacts(data, mjx_model, internal_state)
        contact_filtered_missing_lower_feet_contacts = jnp.mean(feet_floor_contacts * missing_lower_feet_contacts)
        foot_flat_contact_reward = curriculum_coeff * self.foot_flat_contact_coeff * -contact_filtered_missing_lower_feet_contacts

        # Total reward
        tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
        reward_penalty = z_velocity_reward + imu_acceleration_reward + angular_velocity_reward + angular_position_reward + \
                         actuator_joint_nominal_diff_reward +  joint_position_limit_reward + joint_velocity_limit_reward + joint_velocity_reward + \
                         acceleration_reward + torque_reward + power_draw_penalty_reward + action_rate_reward + action_smoothness_reward + \
                         collision_reward + base_height_reward + foot_air_time_reward + symmetry_air_reward + foot_slip_reward + foot_z_velocity_reward + foot_flat_contact_reward
        reward = tracking_reward + reward_penalty + alive_clipped_reward
        reward = jnp.maximum(reward, 0.0) + alive_unclipped_reward
        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Info
        info[f"reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        info[f"reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        info[f"reward/alive_clipped"] = alive_clipped_reward
        info[f"reward/alive_unclipped"] = alive_unclipped_reward
        info[f"reward/z_velocity"] = z_velocity_reward
        info[f"reward/imu_acceleration"] = imu_acceleration_reward
        info[f"reward/angular_velocity"] = angular_velocity_reward
        info[f"reward/angular_position"] = angular_position_reward
        info[f"reward/actuator_joint_nominal_diff"] = actuator_joint_nominal_diff_reward
        info[f"reward/joint_position_limit"] = joint_position_limit_reward
        info[f"reward/joint_velocity_limit"] = joint_velocity_limit_reward
        info[f"reward/joint_velocity"] = joint_velocity_reward
        info[f"reward/joint_acceleration"] = acceleration_reward
        info[f"reward/joint_torque"] = torque_reward
        info[f"reward/power_draw_penalty"] = power_draw_penalty_reward
        info[f"reward/action_rate"] = action_rate_reward
        info[f"reward/action_smoothness"] = action_smoothness_reward
        info[f"reward/collision"] = collision_reward
        info[f"reward/base_height"] = base_height_reward
        info[f"reward/foot_air_time"] = foot_air_time_reward
        info[f"reward/symmetry_air"] = symmetry_air_reward
        info[f"reward/foot_slip"] = foot_slip_reward
        info[f"reward/foot_z_velocity"] = foot_z_velocity_reward
        info[f"reward/foot_flat_contact"] = foot_flat_contact_reward
        info[f"reward/total"] = reward
        info[f"env_info/xy_vel_diff_abs"] = jnp.nan_to_num(jnp.mean(jnp.minimum(jnp.abs(xy_difference), 2*internal_state["max_command_velocity"])), nan=2*internal_state["max_command_velocity"], posinf=2*internal_state["max_command_velocity"], neginf=2*internal_state["max_command_velocity"])

        return reward
