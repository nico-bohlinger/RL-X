import jax.numpy as jnp


class PDControl:
    def __init__(self, env):
        self.env = env
        self.proportional_gains = jnp.asarray(env.initial_mj_model.actuator_user[:, 0], dtype=jnp.float32)
        self.derivative_gains = jnp.asarray(env.initial_mj_model.actuator_user[:, 1], dtype=jnp.float32)
        self.effort_limits = jnp.asarray(env.initial_mj_model.actuator_forcerange[:, 1], dtype=jnp.float32)
        self.action_scales = env.env_config["control"]["pd"]["action_scale"] * self.effort_limits / self.proportional_gains


    def process_action(self, action, internal_state):
        target_joint_positions = self.env.actuator_joint_nominal_positions + self.action_scales * action
        noisy_target_joint_positions = target_joint_positions + internal_state["joint_position_bias"]

        return noisy_target_joint_positions


    def control(self, data, target_joint_positions, proportional_gain_factors, derivative_gain_factors):
        torque = self.proportional_gains * proportional_gain_factors * (target_joint_positions - data.qpos[..., self.env.actuator_joint_mask_qpos]) - self.derivative_gains * derivative_gain_factors * data.qvel[..., self.env.actuator_joint_mask_qvel]
        control = jnp.clip(torque, -self.effort_limits, self.effort_limits)

        return control
