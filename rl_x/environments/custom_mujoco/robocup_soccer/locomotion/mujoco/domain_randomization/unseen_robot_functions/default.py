import numpy as np


class DefaultDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env

        self.mass_inertia_factor = env.env_config["domain_randomization"]["unseen_robot"]["mass_inertia_factor"]
        self.com_factor = env.env_config["domain_randomization"]["unseen_robot"]["com_factor"]
        self.body_position_factor = env.env_config["domain_randomization"]["unseen_robot"]["body_position_factor"]
        self.joint_damping_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_damping_factor"]
        self.joint_armature_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_armature_factor"]
        self.joint_stiffness_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_stiffness_factor"]
        self.joint_friction_loss_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_friction_loss_factor"]
        self.p_gain_factor = env.env_config["domain_randomization"]["unseen_robot"]["p_gain_factor"]
        self.d_gain_factor = env.env_config["domain_randomization"]["unseen_robot"]["d_gain_factor"]
        self.position_offset = env.env_config["domain_randomization"]["unseen_robot"]["position_offset"]

        self.nr_bodies = self.env.initial_mj_model.body_mass[1:].shape[0]
        self.nr_joint_dampings = self.env.initial_mj_model.dof_damping[6:].shape[0]
        self.nr_joint_armatures = self.env.initial_mj_model.dof_armature[6:].shape[0]
        self.nr_joint_stiffnesses = self.env.initial_mj_model.jnt_stiffness[1:].shape[0]
        self.nr_joint_frictionlosses = self.env.initial_mj_model.dof_frictionloss[6:].shape[0]


    def init(self):
        self.env.internal_state["mass_inertia_noise_factors"] = np.ones(self.nr_bodies)
        self.env.internal_state["com_noise_factors"] = np.ones((self.nr_bodies, 3))
        self.env.internal_state["body_position_noise_factors"] = np.ones((self.nr_bodies, 3))
        self.env.internal_state["joint_damping_noise_factors"] = np.ones(self.nr_joint_dampings)
        self.env.internal_state["joint_armature_noise_factors"] = np.ones(self.nr_joint_armatures)
        self.env.internal_state["joint_stiffness_noise_factors"] = np.ones(self.nr_joint_stiffnesses)
        self.env.internal_state["joint_friction_loss_noise_factors"] = np.ones(self.nr_joint_frictionlosses)
        self.env.internal_state["p_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["d_gain_noise_factors"] = np.ones(self.env.nr_actuator_joints)
        self.env.internal_state["position_offsets"] = np.zeros(self.env.nr_actuator_joints)


    def sample(self):
        self.env.internal_state["mass_inertia_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.mass_inertia_factor, high=self.mass_inertia_factor, size=(self.nr_bodies,))
        self.env.internal_state["com_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.com_factor, high=self.com_factor, size=(self.nr_bodies, 3))
        self.env.internal_state["body_position_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.body_position_factor, high=self.body_position_factor, size=(self.nr_bodies, 3))
        self.env.internal_state["joint_damping_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.joint_damping_factor, high=self.joint_damping_factor, size=(self.nr_joint_dampings,))
        self.env.internal_state["joint_armature_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.joint_armature_factor, high=self.joint_armature_factor, size=(self.nr_joint_armatures,))
        self.env.internal_state["joint_stiffness_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.joint_stiffness_factor, high=self.joint_stiffness_factor, size=(self.nr_joint_stiffnesses,))
        self.env.internal_state["joint_friction_loss_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.joint_friction_loss_factor, high=self.joint_friction_loss_factor, size=(self.nr_joint_frictionlosses,))
        self.env.internal_state["p_gain_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.p_gain_factor, high=self.p_gain_factor, size=(self.env.nr_actuator_joints,))
        self.env.internal_state["d_gain_noise_factors"] = 1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.d_gain_factor, high=self.d_gain_factor, size=(self.env.nr_actuator_joints,))
        self.env.internal_state["position_offsets"] = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.position_offset, high=self.position_offset, size=(self.env.nr_actuator_joints,))
