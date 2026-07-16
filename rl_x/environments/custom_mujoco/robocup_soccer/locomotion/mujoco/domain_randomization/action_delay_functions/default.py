import numpy as np


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        self.min_delay_substeps = round(env.env_config["domain_randomization"]["action_delay"]["min_delay_s"] / env.env_config["timestep"])
        self.max_delay_substeps = round(env.env_config["domain_randomization"]["action_delay"]["max_delay_s"] / env.env_config["timestep"])
        self.buffer_length = self.max_delay_substeps + 1


    def init(self):
        self.env.internal_state["action_delay_buffer"] = np.zeros((self.buffer_length, self.env.nr_actuator_joints))
        self.env.internal_state["action_delay_buffer_ptr"] = 0
        self.env.internal_state["action_delay_steps"] = self.min_delay_substeps


    def setup(self):
        self.env.internal_state["action_delay_buffer"] = np.zeros((self.buffer_length, self.env.nr_actuator_joints))
        self.env.internal_state["action_delay_buffer_ptr"] = 0


    def sample(self):
        effective_max_delay_substeps = self.min_delay_substeps + int(self.env.internal_state["env_curriculum_coeff"] * (self.max_delay_substeps - self.min_delay_substeps))
        self.env.internal_state["action_delay_steps"] = self.env.np_rng.integers(self.min_delay_substeps, effective_max_delay_substeps + 1)


    def delay_action(self, action):
        buffer = self.env.internal_state["action_delay_buffer"]
        buffer_ptr = self.env.internal_state["action_delay_buffer_ptr"]
        delay_steps = self.env.internal_state["action_delay_steps"]

        substep_indices = np.arange(self.env.nr_substeps)
        read_indices = (buffer_ptr + substep_indices - delay_steps) % self.buffer_length
        delayed_actions = np.where((substep_indices >= delay_steps).reshape(-1, 1), action, buffer[read_indices])

        write_indices = (buffer_ptr + substep_indices) % self.buffer_length
        buffer[write_indices] = action
        self.env.internal_state["action_delay_buffer_ptr"] = (buffer_ptr + self.env.nr_substeps) % self.buffer_length

        return delayed_actions
