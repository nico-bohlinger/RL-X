import numpy as np


class PDControl:
    def __init__(self, env):
        self.env = env
        self.proportional_gains = np.asarray(env.initial_mj_model.actuator_user[:, 0], dtype=np.float32)
        self.derivative_gains = np.asarray(env.initial_mj_model.actuator_user[:, 1], dtype=np.float32)
        self.effort_limits = np.asarray(env.initial_mj_model.actuator_forcerange[:, 1], dtype=np.float32)
        self.action_scales = env.env_config["control"]["pd"]["action_scale"] * self.effort_limits / self.proportional_gains


    def process_action(self, action):
        target_joint_positions = self.env.actuator_joint_nominal_positions + self.action_scales * action
        noisy_target_joint_positions = target_joint_positions + self.env.internal_state["joint_position_bias"]

        return noisy_target_joint_positions


    def control(self, target_joint_positions):
        torque = self.proportional_gains * self.env.internal_state["proportional_gain_factors"] * (target_joint_positions - self.env.internal_state["data"].qpos[self.env.actuator_joint_mask_qpos]) - self.derivative_gains * self.env.internal_state["derivative_gain_factors"] * self.env.internal_state["data"].qvel[self.env.actuator_joint_mask_qvel]
        control = np.clip(torque, -self.effort_limits, self.effort_limits)

        return control
