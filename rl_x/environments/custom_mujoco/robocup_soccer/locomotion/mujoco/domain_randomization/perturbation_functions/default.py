import numpy as np


class DefaultDRPerturbation:
    def __init__(self, env):
        self.env = env

        self.trunk_velocity_clip_mass_factor = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_clip_mass_factor"]
        self.trunk_velocity_clip_limit = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_clip_limit"]
        self.trunk_velocity_add_chance = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_add_chance"]
        self.max_joint_velocity = env.env_config["domain_randomization"]["perturbation"]["max_joint_velocity"]
        self.max_joint_position = env.env_config["domain_randomization"]["perturbation"]["max_joint_position"]


    def sample(self):
        max_trunk_velocity = np.minimum(np.sum(self.env.internal_state["mj_model"].body_mass) * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)

        trunk_velocity_perturbation = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(6,), low=-max_trunk_velocity, high=max_trunk_velocity)
        trunk_velocity_perturbation = np.where(self.env.np_rng.uniform() < self.trunk_velocity_add_chance, self.env.internal_state["data"].qvel[0:6] + trunk_velocity_perturbation, (trunk_velocity_perturbation * self.env.internal_state["env_curriculum_coeff"]) + (self.env.internal_state["data"].qvel[0:6] * (1 - self.env.internal_state["env_curriculum_coeff"])))
        trunk_velocity = trunk_velocity_perturbation

        joint_velocity = self.env.internal_state["data"].qvel[6:] + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(self.env.internal_state["data"].qvel[6:].shape[0],), low=-self.max_joint_velocity, high=self.max_joint_velocity)
        joint_position = self.env.internal_state["data"].qpos[7:] + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(size=(self.env.internal_state["data"].qpos[7:].shape[0],), low=-self.max_joint_position, high=self.max_joint_position)

        self.env.internal_state["data"].qpos = np.concatenate([self.env.internal_state["data"].qpos[0:7], joint_position])
        self.env.internal_state["data"].qvel = np.concatenate([trunk_velocity, joint_velocity])
