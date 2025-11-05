import jax.numpy as jnp


class PDControl:
    def __init__(self, env, control_frequency_hz=50):
        self.env = env
        self.control_frequency_hz = control_frequency_hz


    def process_action(self, action, internal_state):
        scaled_action = action * internal_state["scaling_factor"]
        target_joint_positions = internal_state["actuator_joint_nominal_positions"] + scaled_action
        noisy_target_joint_positions = target_joint_positions + internal_state["position_offsets"]
        
        return noisy_target_joint_positions
