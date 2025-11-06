import numpy as np
import jax
import jax.numpy as jnp


class RandomCommands:
    def __init__(self, env):
        self.env = env
        self.max_velocity_per_m_factor = self.env.env_config["command"]["max_velocity_per_m_factor"]
        self.clip_max_velocity = self.env.env_config["command"]["clip_max_velocity"]
        self.zero_clip_threshold_percentage = self.env.env_config["command"]["zero_clip_threshold_percentage"]
        self.all_zero_chance = self.env.env_config["command"]["all_zero_chance"]
        self.single_zero_chance = self.env.env_config["command"]["single_zero_chance"]

        self.default_actuator_joint_keep_nominal = np.zeros(env.nr_actuator_joints, dtype=bool)
        self.default_actuator_joint_keep_nominal[env.robot_config["actuator_joints_to_stay_near_nominal"]] = 1.0
        self.default_actuator_joint_keep_nominal = jnp.array(self.default_actuator_joint_keep_nominal)


    def init(self, internal_state):
        internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self, internal_state, should_sample_commands, subkey):
        velocity_sampling_key, all_zeroing_key, single_zeroing_key = jax.random.split(subkey, 3)

        goal_velocities = jax.random.uniform(velocity_sampling_key, (3,), minval=-internal_state["max_command_velocity"], maxval=internal_state["max_command_velocity"])
        goal_velocities = jnp.where(jnp.abs(goal_velocities) < (self.zero_clip_threshold_percentage * internal_state["max_command_velocity"]), 0.0, goal_velocities)
        goal_velocities = jnp.where(jax.random.bernoulli(all_zeroing_key, self.all_zero_chance), jnp.zeros(3), goal_velocities)
        goal_velocities = jnp.where(jax.random.uniform(single_zeroing_key, (3,)) < self.single_zero_chance, 0.0, goal_velocities)

        internal_state["goal_velocities"] = jnp.where(should_sample_commands, goal_velocities, internal_state["goal_velocities"])

        actuator_joint_keep_nominal = jnp.where(jnp.all(goal_velocities == 0.0), jnp.ones(self.env.nr_actuator_joints, dtype=bool), self.default_actuator_joint_keep_nominal)

        internal_state["actuator_joint_keep_nominal"] = jnp.where(should_sample_commands, actuator_joint_keep_nominal, internal_state["actuator_joint_keep_nominal"])
