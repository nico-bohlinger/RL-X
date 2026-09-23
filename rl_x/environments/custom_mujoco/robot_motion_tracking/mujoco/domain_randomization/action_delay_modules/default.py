import numpy as np


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        self.delay_steps_range = env.env_config["domain_randomization"]["action_delay"]["default"]["delay_steps_range"]
        self.enable_randomization = env.env_config["domain_randomization"]["action_delay"]["default"]["enable_randomization"]
        self.history_length = self.delay_steps_range[1] + 1


    def init(self):
        self.env.internal_state["action_delay"] = 0
        self.setup()


    def setup(self):
        self.env.internal_state["action_history"] = np.zeros((self.history_length, self.env.nr_actuators), dtype=np.float32)


    def sample(self):
        delay = self.env.np_rng.integers(self.delay_steps_range[0], self.delay_steps_range[1] + 1)
        self.env.internal_state["action_delay"] = delay if self.enable_randomization and not self.env.internal_state["in_eval_mode"] else 0


    def delay_action(self, action):
        self.env.internal_state["action_history"] = np.concatenate((np.clip(action, -self.env.env_config["action_clip"], self.env.env_config["action_clip"])[None], self.env.internal_state["action_history"][:-1]), axis=0)

        return self.env.internal_state["action_history"][self.env.internal_state["action_delay"]]
