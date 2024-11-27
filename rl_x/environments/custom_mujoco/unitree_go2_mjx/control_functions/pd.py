import jax.numpy as jnp


class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz
        self.p_gain = env.robot_config["p_gain"]
        self.d_gain = env.robot_config["d_gain"]
        self.scaling_factor = env.robot_config["scaling_factor"]


    def init(self, internal_state):
        internal_state["extrinsic_p_gain_factor"] = jnp.ones(self.env.initial_mjx_model.nu)
        internal_state["extrinsic_d_gain_factor"] = jnp.ones(self.env.initial_mjx_model.nu)
        internal_state["extrinsic_motor_strength"] = jnp.ones(self.env.initial_mjx_model.nu)
        internal_state["extrinsic_position_offset"] = jnp.zeros(self.env.initial_mjx_model.nu)


    def process_action(self, action, internal_state, data):
        scaled_action = action * self.scaling_factor
        target_joint_positions = self.env.joint_nominal_positions + scaled_action
        torques = self.p_gain * internal_state["extrinsic_p_gain_factor"] * (target_joint_positions - data.qpos[7:] + internal_state["extrinsic_position_offset"]) \
                  - self.d_gain * internal_state["extrinsic_d_gain_factor"] * data.qvel[6:]
        
        return torques * internal_state["extrinsic_motor_strength"]
