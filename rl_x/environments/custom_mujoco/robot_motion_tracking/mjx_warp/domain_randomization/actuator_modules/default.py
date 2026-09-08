import jax
import jax.numpy as jnp


class DefaultActuator:
    def __init__(self, env):
        self.env = env

        self.joint_bias_range = env.env_config["domain_randomization"]["actuator"]["default"]["joint_bias_range"]
        self.enable_randomization = env.env_config["domain_randomization"]["actuator"]["default"]["enable_randomization"]
        self.enable_joint_bias = env.env_config["domain_randomization"]["actuator"]["default"]["enable_joint_bias"]
        self.enable_gain_randomization = env.env_config["domain_randomization"]["actuator"]["default"]["enable_gain_randomization"]
        self.proportional_gain_range = env.env_config["domain_randomization"]["actuator"]["default"]["proportional_gain_range"]
        self.derivative_gain_range = env.env_config["domain_randomization"]["actuator"]["default"]["derivative_gain_range"]


    def init(self, internal_state):
        internal_state["joint_position_bias"] = jnp.zeros((self.env.nr_envs, self.env.nr_actuators), jnp.float32)
        internal_state["proportional_gain_factors"] = jnp.ones((self.env.nr_envs, self.env.nr_actuators), jnp.float32)
        internal_state["derivative_gain_factors"] = jnp.ones((self.env.nr_envs, self.env.nr_actuators), jnp.float32)


    def sample_bias(self, internal_state, should_randomize, key):
        bias = jax.random.uniform(key, (self.env.nr_envs, self.env.nr_actuators), minval=self.joint_bias_range[0], maxval=self.joint_bias_range[1])
        bias = jnp.where(self.enable_randomization & self.enable_joint_bias & ~internal_state["in_eval_mode"], bias, 0.0)
        internal_state["joint_position_bias"] = jnp.where(should_randomize[..., None], bias, internal_state["joint_position_bias"])


    def sample_gains(self, internal_state, should_randomize, key):
        key_p, key_d = jax.random.split(key)
        enabled = self.enable_randomization & self.enable_gain_randomization & ~internal_state["in_eval_mode"]
        proportional = jnp.where(enabled, jax.random.uniform(key_p, (self.env.nr_envs, self.env.nr_actuators), minval=self.proportional_gain_range[0], maxval=self.proportional_gain_range[1]), 1.0)
        derivative = jnp.where(enabled, jax.random.uniform(key_d, (self.env.nr_envs, self.env.nr_actuators), minval=self.derivative_gain_range[0], maxval=self.derivative_gain_range[1]), 1.0)
        internal_state["proportional_gain_factors"] = jnp.where(should_randomize[..., None], proportional, internal_state["proportional_gain_factors"])
        internal_state["derivative_gain_factors"] = jnp.where(should_randomize[..., None], derivative, internal_state["derivative_gain_factors"])
