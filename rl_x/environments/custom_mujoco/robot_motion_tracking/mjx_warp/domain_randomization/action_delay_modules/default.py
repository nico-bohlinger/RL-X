import jax
import jax.numpy as jnp


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        self.delay_steps_range = env.env_config["domain_randomization"]["action_delay"]["default"]["delay_steps_range"]
        self.enable_randomization = env.env_config["domain_randomization"]["action_delay"]["default"]["enable_randomization"]
        self.history_length = self.delay_steps_range[1] + 1


    def init(self, internal_state):
        internal_state["action_history"] = jnp.zeros((self.env.nr_envs, self.history_length, self.env.nr_actuators), jnp.float32)
        internal_state["action_delay"] = jnp.zeros(self.env.nr_envs, jnp.int32)


    def setup(self, internal_state, is_episode_start):
        internal_state["action_history"] = jnp.where(is_episode_start[:, None, None], 0.0, internal_state["action_history"])


    def sample(self, internal_state, should_randomize, key):
        delay = jax.random.randint(key, (self.env.nr_envs,), self.delay_steps_range[0], self.delay_steps_range[1] + 1)
        delay = jnp.where(self.enable_randomization & ~internal_state["in_eval_mode"], delay, 0)
        internal_state["action_delay"] = jnp.where(should_randomize, delay, internal_state["action_delay"])


    def delay_action(self, action, internal_state):
        internal_state["action_history"] = jnp.concatenate((jnp.clip(action, -self.env.env_config["action_clip"], self.env.env_config["action_clip"])[:, None], internal_state["action_history"][:, :-1]), axis=1)

        return internal_state["action_history"][jnp.arange(self.env.nr_envs), internal_state["action_delay"]]
