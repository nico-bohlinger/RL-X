import jax
import jax.numpy as jnp


class DefaultActionDelay:
    def __init__(self, env):
        self.env = env

        self.min_delay_substeps = round(env.env_config["domain_randomization"]["action_delay"]["min_delay_s"] / env.env_config["timestep"])
        self.max_delay_substeps = round(env.env_config["domain_randomization"]["action_delay"]["max_delay_s"] / env.env_config["timestep"])
        self.buffer_length = self.max_delay_substeps + 1


    def init(self, internal_state):
        internal_state["action_delay_buffer"] = jnp.zeros((self.env.nr_envs, self.buffer_length, self.env.nr_actuator_joints))
        internal_state["action_delay_buffer_ptr"] = jnp.array(0, dtype=jnp.int32)
        internal_state["action_delay_steps"] = jnp.full(self.env.nr_envs, self.min_delay_substeps, dtype=jnp.int32)


    def setup(self, internal_state):
        internal_state["action_delay_buffer"] = jnp.zeros((self.env.nr_envs, self.buffer_length, self.env.nr_actuator_joints))
        internal_state["action_delay_buffer_ptr"] = jnp.array(0, dtype=jnp.int32)


    def sample(self, internal_state, should_randomize, key):
        effective_max_delay_substeps = self.min_delay_substeps + jnp.floor(internal_state["env_curriculum_coeff"] * (self.max_delay_substeps - self.min_delay_substeps)).astype(jnp.int32)
        sampled_delay_steps = self.min_delay_substeps + jnp.floor(jax.random.uniform(key, (self.env.nr_envs,)) * (effective_max_delay_substeps - self.min_delay_substeps + 1)).astype(jnp.int32)
        internal_state["action_delay_steps"] = jnp.where(should_randomize, sampled_delay_steps, internal_state["action_delay_steps"])


    def delay_action(self, action, internal_state):
        buffer = internal_state["action_delay_buffer"]
        buffer_ptr = internal_state["action_delay_buffer_ptr"]
        delay_steps = internal_state["action_delay_steps"]

        substep_indices = jnp.arange(self.env.nr_substeps)
        read_indices = (buffer_ptr + substep_indices[None] - delay_steps[:, None]) % self.buffer_length
        buffered_actions = jnp.take_along_axis(buffer, read_indices[..., None], axis=1)
        delayed_actions = jnp.where((substep_indices[None] >= delay_steps[:, None])[..., None], action[:, None], buffered_actions)

        write_indices = (buffer_ptr + substep_indices) % self.buffer_length
        internal_state["action_delay_buffer"] = buffer.at[:, write_indices].set(action[:, None])
        internal_state["action_delay_buffer_ptr"] = (buffer_ptr + self.env.nr_substeps) % self.buffer_length

        return jnp.swapaxes(delayed_actions, 0, 1)
