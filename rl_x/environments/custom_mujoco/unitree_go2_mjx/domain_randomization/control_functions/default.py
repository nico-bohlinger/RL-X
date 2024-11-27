import jax
import jax.numpy as jnp


class DefaultDRControl:
    def __init__(self, env):
        self.env = env

        self.motor_strength_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["motor_strength_min"]
        self.motor_strength_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["motor_strength_max"]
        self.p_gain_factor_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["p_gain_factor_min"]
        self.p_gain_factor_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["p_gain_factor_max"]
        self.d_gain_factor_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["d_gain_factor_min"]
        self.d_gain_factor_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["d_gain_factor_max"]
        self.asymmetric_factor_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["asymmetric_factor_min"]
        self.asymmetric_factor_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["asymmetric_factor_max"]
        self.position_offset_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["position_offset_min"]
        self.position_offset_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["control"]["position_offset_max"]


    def sample(self, internal_state, should_randomize, key):
        keys = jax.random.split(key, 7)

        sampled_motor_strength = jax.random.uniform(keys[0], (1,), minval=self.motor_strength_min, maxval=self.motor_strength_max)
        sampled_asymmetric_factor_m = jax.random.uniform(keys[1], (self.env.initial_mjx_model.nu,), minval=self.asymmetric_factor_min, maxval=self.asymmetric_factor_max)
        internal_state["extrinsic_motor_strength"] = jnp.where(should_randomize, sampled_motor_strength * sampled_asymmetric_factor_m, internal_state["extrinsic_motor_strength"])

        sampled_p_gain_factor = jax.random.uniform(keys[2], (1,), minval=self.p_gain_factor_min, maxval=self.p_gain_factor_max)
        sampled_asymmetric_factor_p = jax.random.uniform(keys[3], (self.env.initial_mjx_model.nu,), minval=self.asymmetric_factor_min, maxval=self.asymmetric_factor_max)
        internal_state["extrinsic_p_gain_factor"] = jnp.where(should_randomize, sampled_p_gain_factor * sampled_asymmetric_factor_p, internal_state["extrinsic_p_gain_factor"])

        sampled_d_gain_factor = jax.random.uniform(keys[4], (1,), minval=self.d_gain_factor_min, maxval=self.d_gain_factor_max)
        sampled_asymmetric_factor_d = jax.random.uniform(keys[5], (self.env.initial_mjx_model.nu,), minval=self.asymmetric_factor_min, maxval=self.asymmetric_factor_max)
        internal_state["extrinsic_d_gain_factor"] = jnp.where(should_randomize, sampled_d_gain_factor * sampled_asymmetric_factor_d, internal_state["extrinsic_d_gain_factor"])

        sampled_position_offset = jax.random.uniform(keys[6], (self.env.initial_mjx_model.nu,), minval=self.position_offset_min, maxval=self.position_offset_max)
        internal_state["extrinsic_position_offset"] = jnp.where(should_randomize, sampled_position_offset, internal_state["extrinsic_position_offset"])
