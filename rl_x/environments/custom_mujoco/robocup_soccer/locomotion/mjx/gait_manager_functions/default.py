import jax
import jax.numpy as jnp


class DefaultGaitManager:
    def __init__(self, env):
        self.env = env

        self.gait_period = env.env_config["gait_manager"]["gait_period"]
        self.gait_period_randomization_width = env.env_config["gait_manager"]["gait_period_randomization_width"]

        self.gait_mean_freq = 1.0 / self.gait_period
        self.gait_stand_phase_value = jnp.pi


    def init(self, internal_state):
        internal_state["gait_phase_offset"] = jnp.array([0.0, -jnp.pi])
        internal_state["gait_phase"] = internal_state["gait_phase_offset"]
        internal_state["gait_freq"] = self.gait_mean_freq
        internal_state["gait_phase_dt"] = (2.0 * jnp.pi * self.env.dt) * internal_state["gait_freq"]


    def setup(self, internal_state, key):
        random_phase_key, random_frequency_key = jax.random.split(key)
        random_phase_0 = internal_state["env_curriculum_coeff"] * jax.random.uniform(random_phase_key, minval=-jnp.pi, maxval=jnp.pi)
        random_phase_offsets = jnp.array([random_phase_0, self.wrap_to_pi(random_phase_0 + jnp.pi)])
        gait_phase_offset = jnp.where(internal_state["in_eval_mode"], jnp.array([0.0, -jnp.pi]), random_phase_offsets)

        low = self.gait_mean_freq - (internal_state["env_curriculum_coeff"] * self.gait_period_randomization_width)
        high = self.gait_mean_freq + (internal_state["env_curriculum_coeff"] * self.gait_period_randomization_width)
        random_gait_frequency = jax.random.uniform(random_frequency_key, minval=low, maxval=high)
        gait_freq = jnp.where(internal_state["in_eval_mode"], self.gait_mean_freq, random_gait_frequency)

        internal_state["gait_phase_offset"] = gait_phase_offset
        internal_state["gait_phase"] = gait_phase_offset
        internal_state["gait_freq"] = gait_freq
        internal_state["gait_phase_dt"] = (2.0 * jnp.pi * self.env.dt) * gait_freq


    def wrap_to_pi(self, x):
        return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


    def get_phase_features(self, internal_state):
        phase_tp1 = self.wrap_to_pi(internal_state["gait_phase"] + internal_state["gait_phase_dt"])
        return jnp.concatenate([jnp.sin(phase_tp1), jnp.cos(phase_tp1)], axis=-1)


    def get_phase_for_reward(self, internal_state):
        phase_tp1 = self.wrap_to_pi(internal_state["gait_phase"] + internal_state["gait_phase_dt"])
        is_standing_command = jnp.all(internal_state["goal_velocities"] == 0.0)
        return jnp.where(is_standing_command, jnp.full((2,), self.gait_stand_phase_value), phase_tp1)


    def step(self, internal_state):
        internal_state["gait_phase"] = self.wrap_to_pi(internal_state["gait_phase"] + internal_state["gait_phase_dt"])
