import numpy as np


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
        self.default_actuator_joint_keep_nominal = np.array(self.default_actuator_joint_keep_nominal)


    def init(self):
        self.env.internal_state["actuator_joint_keep_nominal"] = self.default_actuator_joint_keep_nominal


    def get_next_command(self):
        goal_velocities = self.env.np_rng.uniform(size=(3,), low=-self.env.internal_state["max_command_velocity"], high=self.env.internal_state["max_command_velocity"])
        goal_velocities = np.where(np.abs(goal_velocities) < (self.zero_clip_threshold_percentage * self.env.internal_state["max_command_velocity"]), 0.0, goal_velocities)
        goal_velocities = np.where(self.env.np_rng.binomial(n=1, p=self.all_zero_chance), np.zeros(3), goal_velocities)
        goal_velocities = np.where(self.env.np_rng.uniform(size=(3,)) < self.single_zero_chance, 0.0, goal_velocities)

        self.env.internal_state["goal_velocities"] = goal_velocities
