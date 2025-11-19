import numpy as np


class DefaultDRJointDropout:
    def __init__(self, env):
        self.env = env
        
        self.dropout_open_chance = env.env_config["domain_randomization"]["joint_dropout"]["dropout_open_chance"]
        self.dropout_lock_chance = env.env_config["domain_randomization"]["joint_dropout"]["dropout_lock_chance"]


    def init(self):
        self.env.internal_state["joint_dropout_open_mask"] = np.ones(self.env.nr_actuator_joints, dtype=bool)
        self.env.internal_state["joint_dropout_lock_mask"] = np.ones(self.env.nr_actuator_joints, dtype=bool)


    def sample(self):
        self.env.internal_state["joint_dropout_open_mask"] = self.env.np_rng.binomial(n=1, p=(1 - self.env.internal_state["env_curriculum_coeff"] * self.dropout_open_chance), size=self.env.internal_state["joint_dropout_open_mask"].shape) == 1.0
        self.env.internal_state["joint_dropout_lock_mask"] = self.env.np_rng.binomial(n=1, p=(1 - self.env.internal_state["env_curriculum_coeff"] * self.dropout_lock_chance), size=self.env.internal_state["joint_dropout_lock_mask"].shape) == 1.0
        self.env.internal_state["joint_dropout_mask"] = self.env.internal_state["joint_dropout_open_mask"] | self.env.internal_state["joint_dropout_lock_mask"]

        modified_actuator_gainprm = self.env.internal_state["partial_actuator_gainprm_without_dropout"] * self.env.internal_state["joint_dropout_mask"]
        modified_actuator_biasprm = self.env.internal_state["partial_actuator_biasprm_without_dropout"] * self.env.internal_state["joint_dropout_mask"].reshape(-1, 1)

        self.env.internal_state["mj_model"].actuator_gainprm[:, 0] = modified_actuator_gainprm
        self.env.internal_state["mj_model"].actuator_biasprm[:, 1:3] = modified_actuator_biasprm

        locked_actuator_joint_ranges_min = self.env.internal_state["actuator_joint_nominal_positions"] - 0.001
        locked_actuator_joint_ranges_max = self.env.internal_state["actuator_joint_nominal_positions"] + 0.001
        locked_actuator_joint_ranges = np.stack([locked_actuator_joint_ranges_min, locked_actuator_joint_ranges_max], axis=-1)
        actuator_joint_ranges = np.where(
            self.env.internal_state["joint_dropout_lock_mask"].reshape(-1, 1),
            self.env.internal_state["seen_joint_ranges"][self.env.actuator_joint_mask_joints - 1],
            locked_actuator_joint_ranges
        )
        self.env.internal_state["mj_model"].jnt_range[self.env.actuator_joint_mask_joints] = actuator_joint_ranges
