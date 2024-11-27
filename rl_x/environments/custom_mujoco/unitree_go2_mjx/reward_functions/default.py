import jax.numpy as jnp


class DefaultReward:
    def __init__(self, env,
                 xy_tracking_temperature=0.25, yaw_tracking_temperature=0.25,
                 soft_joint_position_limit=0.9, air_time_max=0.5):
        self.env = env
        self.xy_tracking_temperature = xy_tracking_temperature
        self.yaw_tracking_temperature = yaw_tracking_temperature
        self.soft_joint_position_limit = soft_joint_position_limit
        self.air_time_max = air_time_max

        self.tracking_xy_velocity_command_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["tracking_xy_velocity_command_coeff"] * env.dt
        self.tracking_yaw_velocity_command_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["tracking_yaw_velocity_command_coeff"] * env.dt
        self.curriculum_steps = env.robot_config["locomotion_envs"]["default"]["reward"]["curriculum_steps"]
        self.z_velocity_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["z_velocity_coeff"] * env.dt
        self.pitch_roll_vel_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["pitch_roll_vel_coeff"] * env.dt
        self.pitch_roll_pos_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["pitch_roll_pos_coeff"] * env.dt
        self.joint_nominal_diff_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["joint_nominal_diff_coeff"] * env.dt
        self.joint_position_limit_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["joint_position_limit_coeff"] * env.dt
        self.joint_acceleration_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["joint_acceleration_coeff"] * env.dt
        self.joint_torque_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["joint_torque_coeff"] * env.dt
        self.action_rate_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["action_rate_coeff"] * env.dt
        self.collision_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["collision_coeff"] * env.dt
        self.base_height_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["base_height_coeff"] * env.dt
        self.air_time_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["air_time_coeff"] * env.dt
        self.symmetry_air_coeff = env.robot_config["locomotion_envs"]["default"]["reward"]["symmetry_air_coeff"] * env.dt

        self.nominal_trunk_z = env.initial_mj_model.keyframe("home").qpos[2]
        self.feet_symmetry_pairs = env.feet_symmetry_pairs
        self.feet_touchdown_names = [f"time_since_last_touchdown_{foot_name.replace('_foot', '').lower()}" for foot_name in env.feet_names]


    def init(self, internal_state, mjx_model):
        internal_state["joint_limits"] = self.calculate_joint_limits(mjx_model)
        self.setup(internal_state)


    def calculate_joint_limits(self, mjx_model):
        joint_limits = mjx_model.jnt_range[1:]
        joint_limits_midpoint = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
        joint_limits_range = joint_limits[:, 1] - joint_limits[:, 0]
        lower_joint_limits = joint_limits_midpoint - joint_limits_range / 2 * self.soft_joint_position_limit
        upper_joint_limits = joint_limits_midpoint + joint_limits_range / 2 * self.soft_joint_position_limit
        return jnp.stack([lower_joint_limits, upper_joint_limits], axis=1)

    
    def handle_model_change(self, internal_state, mjx_model, should_change):
        internal_state["joint_limits"] = jnp.where(should_change, self.calculate_joint_limits(mjx_model), internal_state["joint_limits"])


    def setup(self, internal_state):
        for foot_touchdown_name in self.feet_touchdown_names:
            internal_state[foot_touchdown_name] = 0.0
        internal_state["prev_joint_vel"] = jnp.zeros(self.env.initial_mjx_model.nu)
        internal_state["sum_tracking_performance_percentage"] = 0.0


    def step(self, data, mjx_model, internal_state):
        for foot_touchdown_name, foot_name in zip(self.feet_touchdown_names, self.env.feet_names):
            internal_state[foot_touchdown_name] = jnp.where(self.env.terrain_function.check_foot_floor_contact(data, mjx_model, internal_state, foot_name), 0, internal_state[foot_touchdown_name] + self.env.dt)
        internal_state["prev_joint_vel"] = data.qvel[6:]


    def reward_and_info(self, data, mjx_model, internal_state, action, info):
        curriculum_coeff = jnp.where(internal_state["reached_max_timesteps"], 1.0, jnp.minimum(internal_state["total_timesteps"] / self.curriculum_steps, 1.0))
        
        # Tracking velocity command reward
        current_global_linear_velocity = data.qvel[:3]
        current_local_linear_velocity = internal_state["orientation_rotation_inverse"].apply(current_global_linear_velocity)
        desired_local_linear_velocity_xy = internal_state["goal_velocities"][:2]
        xy_velocity_difference_norm = jnp.sum(jnp.square(desired_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        tracking_xy_velocity_command_reward = self.tracking_xy_velocity_command_coeff * jnp.exp(-xy_velocity_difference_norm / self.xy_tracking_temperature)

        # Tracking angular velocity command reward
        current_local_angular_velocity = data.qvel[3:6]
        desired_local_yaw_velocity = internal_state["goal_velocities"][2]
        yaw_velocity_difference_norm = jnp.sum(jnp.square(current_local_angular_velocity[2] - desired_local_yaw_velocity))
        tracking_yaw_velocity_command_reward = self.tracking_yaw_velocity_command_coeff * jnp.exp(-yaw_velocity_difference_norm / self.yaw_tracking_temperature)

        # Linear velocity reward
        z_velocity_squared = current_local_linear_velocity[2] ** 2
        linear_velocity_reward = curriculum_coeff * self.z_velocity_coeff * -z_velocity_squared

        # Angular velocity reward
        angular_velocity_norm = jnp.sum(jnp.square(current_local_angular_velocity[:2]))
        angular_velocity_reward = curriculum_coeff * self.pitch_roll_vel_coeff * -angular_velocity_norm

        # Angular position reward
        pitch_roll_position_norm = jnp.sum(jnp.square(internal_state["orientation_euler"][:2]))
        angular_position_reward = curriculum_coeff * self.pitch_roll_pos_coeff * -pitch_roll_position_norm

        # Joint nominal position difference reward
        joint_nominal_diff_norm = 0.0
        joint_nominal_diff_reward = curriculum_coeff * self.joint_nominal_diff_coeff * -joint_nominal_diff_norm

        # Joint position limit reward
        joint_positions = data.qpos[7:]
        lower_limit_penalty = -jnp.minimum(joint_positions - internal_state["joint_limits"][:, 0], 0.0).sum()
        upper_limit_penalty = jnp.maximum(joint_positions - internal_state["joint_limits"][:, 1], 0.0).sum()
        joint_position_limit_reward = curriculum_coeff * self.joint_position_limit_coeff * -(lower_limit_penalty + upper_limit_penalty)

        # Joint acceleration reward
        acceleration_norm = jnp.sum(jnp.square((internal_state["prev_joint_vel"] - data.qvel[6:]) / self.env.dt))
        acceleration_reward = curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm

        # Joint torque reward
        torque_norm = jnp.sum(jnp.square(data.qfrc_actuator[6:]))
        torque_reward = curriculum_coeff * self.joint_torque_coeff * -torque_norm

        # Action rate reward
        action_rate_norm = jnp.sum(jnp.square(action - internal_state["last_action"]))
        action_rate_reward = curriculum_coeff * self.action_rate_coeff * -action_rate_norm

        # Collision reward
        nr_collisions = 0
        collision_reward = curriculum_coeff * self.collision_coeff * -nr_collisions

        # Walking height
        trunk_z = internal_state["robot_height_over_ground"]
        height_difference_squared = (trunk_z - self.nominal_trunk_z) ** 2
        base_height_reward = curriculum_coeff * self.base_height_coeff * -height_difference_squared

        # Air time reward
        air_time_reward = 0.0
        foot_floor_contacts = jnp.array([self.env.terrain_function.check_foot_floor_contact(data, mjx_model, internal_state, foot_name) for foot_name in self.env.feet_names])
        for i in range(len(self.feet_touchdown_names)):
            air_time_reward += jnp.where(foot_floor_contacts[i], internal_state[self.feet_touchdown_names[i]] - self.air_time_max, 0.0)
        air_time_reward = curriculum_coeff * self.air_time_coeff * air_time_reward

        # Symmetry reward
        symmetry_air_violations = jnp.sum(jnp.where(~foot_floor_contacts[self.feet_symmetry_pairs[:, 0]] & ~foot_floor_contacts[self.feet_symmetry_pairs[:, 1]], 1, 0))
        symmetry_air_reward = curriculum_coeff * self.symmetry_air_coeff * -symmetry_air_violations

        # Total reward
        tracking_reward = tracking_xy_velocity_command_reward + tracking_yaw_velocity_command_reward
        reward_penalty = linear_velocity_reward + angular_velocity_reward + angular_position_reward + joint_nominal_diff_reward + \
                         joint_position_limit_reward + acceleration_reward + torque_reward + action_rate_reward + \
                         collision_reward + base_height_reward + air_time_reward + symmetry_air_reward
        reward = tracking_reward + reward_penalty
        reward = jnp.maximum(reward, 0.0)

        # Info
        info["reward/track_xy_vel_cmd"] = tracking_xy_velocity_command_reward
        info["reward/track_yaw_vel_cmd"] = tracking_yaw_velocity_command_reward
        info["reward/linear_velocity"] = linear_velocity_reward
        info["reward/angular_velocity"] = angular_velocity_reward
        info["reward/angular_position"] = angular_position_reward
        info["reward/joint_nominal_diff"] = joint_nominal_diff_reward
        info["reward/joint_position_limit"] = joint_position_limit_reward
        info["reward/torque"] = torque_reward
        info["reward/acceleration"] = acceleration_reward
        info["reward/action_rate"] = action_rate_reward
        info["reward/collision"] = collision_reward
        info["reward/base_height"] = base_height_reward
        info["reward/air_time"] = air_time_reward
        info["reward/symmetry_air"] = symmetry_air_reward
        info["env_info/target_x_vel"] = desired_local_linear_velocity_xy[0]
        info["env_info/target_y_vel"] = desired_local_linear_velocity_xy[1]
        info["env_info/target_yaw_vel"] = desired_local_yaw_velocity
        info["env_info/current_x_vel"] = current_local_linear_velocity[0]
        info["env_info/current_y_vel"] = current_local_linear_velocity[1]
        info["env_info/current_yaw_vel"] = current_local_angular_velocity[2]
        info["env_info/symmetry_violations"] = symmetry_air_violations
        info["env_info/walk_height"] = trunk_z
        info["env_info/xy_vel_diff_norm"] = xy_velocity_difference_norm
        info["env_info/yaw_vel_diff_norm"] = yaw_velocity_difference_norm
        info["env_info/torque_norm"] = torque_norm
        info["env_info/acceleration_norm"] = acceleration_norm
        info["env_info/action_rate_norm"] = action_rate_norm

        return reward
