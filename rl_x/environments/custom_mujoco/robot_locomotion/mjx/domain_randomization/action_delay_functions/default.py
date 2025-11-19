import jax
import jax.numpy as jnp


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        self.max_nr_delay_steps = env.env_config["domain_randomization"]["action_delay"]["max_nr_delay_steps"]
        self.mixed_chance = env.env_config["domain_randomization"]["action_delay"]["mixed_chance"]

        self.current_mixed = False
        self.current_nr_delay_steps = 0


    def init(self, internal_state):
        internal_state["action_current_mixed"] = False
        internal_state["action_current_nr_delay_steps"] = 0


    def setup(self, internal_state):
        internal_state["action_history"] = jnp.zeros((self.max_nr_delay_steps + 1, self.env.nr_actuator_joints))


    def sample(self, internal_state, should_randomize, key):
        internal_state["action_current_mixed"] = jnp.where(should_randomize, jax.random.uniform(key) < self.mixed_chance, internal_state["action_current_mixed"])
        internal_state["action_current_nr_delay_steps"] = jnp.where(should_randomize, 0, internal_state["action_current_nr_delay_steps"])


    def delay_action(self, action, internal_state, key):
        current_nr_delay_steps = jnp.ceil(jnp.where(
            internal_state["action_current_mixed"],
            jax.random.randint(key, (1,), 0, self.max_nr_delay_steps+1)[0],
            internal_state["action_current_nr_delay_steps"]
        ) * internal_state["env_curriculum_coeff"]).astype(jnp.int32)

        internal_state["action_history"] = jnp.roll(internal_state["action_history"], -1, axis=0)
        internal_state["action_history"] = internal_state["action_history"].at[-1].set(action)

        chosen_action = internal_state["action_history"][-1-current_nr_delay_steps]

        return chosen_action
