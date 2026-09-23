import numpy as np


class DefaultActuator:
    def __init__(self, env):
        self.env = env

        self.joint_bias_range = env.env_config["domain_randomization"]["actuator"]["default"]["joint_bias_range"]
        self.enable_randomization = env.env_config["domain_randomization"]["actuator"]["default"]["enable_randomization"]
        self.enable_joint_bias = env.env_config["domain_randomization"]["actuator"]["default"]["enable_joint_bias"]
        self.enable_gain_randomization = env.env_config["domain_randomization"]["actuator"]["default"]["enable_gain_randomization"]
        self.proportional_gain_range = env.env_config["domain_randomization"]["actuator"]["default"]["proportional_gain_range"]
        self.derivative_gain_range = env.env_config["domain_randomization"]["actuator"]["default"]["derivative_gain_range"]


    def init(self):
        self.env.internal_state["joint_position_bias"] = np.zeros(self.env.nr_actuators)
        self.env.internal_state["proportional_gain_factors"] = np.ones(self.env.nr_actuators)
        self.env.internal_state["derivative_gain_factors"] = np.ones(self.env.nr_actuators)


    def sample_bias(self):
        bias = self.env.np_rng.uniform(*self.joint_bias_range, self.env.nr_actuators)
        self.env.internal_state["joint_position_bias"] = np.where(self.enable_randomization and self.enable_joint_bias and not self.env.internal_state["in_eval_mode"], bias, 0.0)


    def sample_gains(self):
        enabled = self.enable_randomization and self.enable_gain_randomization and not self.env.internal_state["in_eval_mode"]
        self.env.internal_state["proportional_gain_factors"] = np.where(enabled, self.env.np_rng.uniform(*self.proportional_gain_range, self.env.nr_actuators), 1.0)
        self.env.internal_state["derivative_gain_factors"] = np.where(enabled, self.env.np_rng.uniform(*self.derivative_gain_range, self.env.nr_actuators), 1.0)
