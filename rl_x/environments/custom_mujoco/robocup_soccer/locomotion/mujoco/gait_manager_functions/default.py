import numpy as np


class DefaultGaitManager:
    def __init__(self, env):
        self.env = env

        self.gait_period = env.env_config["gait_manager"]["gait_period"]
        self.gait_period_randomization_width = env.env_config["gait_manager"]["gait_period_randomization_width"]

        self.gait_mean_freq = 1.0 / self.gait_period
        self.gait_stand_phase_value = np.pi


    def init(self):
        self.env.internal_state["gait_phase_offset"] = np.array([0.0, -np.pi])
        self.env.internal_state["gait_phase"] = self.env.internal_state["gait_phase_offset"]
        self.env.internal_state["gait_freq"] = self.gait_mean_freq
        self.env.internal_state["gait_phase_dt"] = (2.0 * np.pi * self.env.dt) * self.env.internal_state["gait_freq"]


    def setup(self):
        random_phase_0 = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-np.pi, high=np.pi)
        random_phase_offsets = np.array([random_phase_0, self.wrap_to_pi(random_phase_0 + np.pi)])
        gait_phase_offset = np.where(self.env.internal_state["in_eval_mode"], np.array([0.0, -np.pi]), random_phase_offsets)

        low = self.gait_mean_freq - (self.env.internal_state["env_curriculum_coeff"] * self.gait_period_randomization_width)
        high = self.gait_mean_freq + (self.env.internal_state["env_curriculum_coeff"] * self.gait_period_randomization_width)
        random_gait_frequency = self.env.np_rng.uniform(low=low, high=high)
        gait_freq = np.where(self.env.internal_state["in_eval_mode"], self.gait_mean_freq, random_gait_frequency)

        self.env.internal_state["gait_phase_offset"] = gait_phase_offset
        self.env.internal_state["gait_phase"] = gait_phase_offset
        self.env.internal_state["gait_freq"] = gait_freq
        self.env.internal_state["gait_phase_dt"] = (2.0 * np.pi * self.env.dt) * gait_freq


    def wrap_to_pi(self, x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi


    def get_phase_features(self):
        phase_tp1 = self.wrap_to_pi(self.env.internal_state["gait_phase"] + self.env.internal_state["gait_phase_dt"])
        return np.concatenate([np.sin(phase_tp1), np.cos(phase_tp1)], axis=-1)


    def get_phase_for_reward(self):
        phase_tp1 = self.wrap_to_pi(self.env.internal_state["gait_phase"] + self.env.internal_state["gait_phase_dt"])
        is_standing_command = np.all(self.env.internal_state["goal_velocities"] == 0.0)
        return np.where(is_standing_command, np.full((2,), self.gait_stand_phase_value), phase_tp1)


    def step(self):
        self.env.internal_state["gait_phase"] = self.wrap_to_pi(self.env.internal_state["gait_phase"] + self.env.internal_state["gait_phase_dt"])
