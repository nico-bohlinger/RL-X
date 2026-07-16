import numpy as np


class DefaultDRMuJoCoModel:
    def __init__(self, env):
        self.env = env

        self.friction_tangential_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_tangential_factor"]
        self.friction_torsional_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_torsional_factor"]
        self.friction_rolling_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_rolling_factor"]
        self.timeconst_log_range = env.env_config["domain_randomization"]["mujoco_model"]["timeconst_log_range"]
        self.dampratio_factor = env.env_config["domain_randomization"]["mujoco_model"]["dampratio_factor"]
        self.foot_solimp_factor = env.env_config["domain_randomization"]["mujoco_model"]["foot_solimp_factor"]
        self.add_impratio = env.env_config["domain_randomization"]["mujoco_model"]["add_impratio"]
        self.xy_gravity = env.env_config["domain_randomization"]["mujoco_model"]["xy_gravity"]
        self.z_gravity_factor = env.env_config["domain_randomization"]["mujoco_model"]["z_gravity_factor"]
        self.density_factor = env.env_config["domain_randomization"]["mujoco_model"]["density_factor"]
        self.viscosity_factor = env.env_config["domain_randomization"]["mujoco_model"]["viscosity_factor"]

        self.default_friction_tangential = self.env.initial_mj_model.geom_friction[self.env.foot_geom_indices, 0]
        self.default_friction_torsional = self.env.initial_mj_model.geom_friction[self.env.foot_geom_indices, 1]
        self.default_friction_rolling = self.env.initial_mj_model.geom_friction[self.env.foot_geom_indices, 2]
        self.default_timeconst = self.env.initial_mj_model.geom_solref[self.env.foot_geom_indices, 0]
        self.default_dampratio = self.env.initial_mj_model.geom_solref[self.env.foot_geom_indices, 1]
        self.min_timeconst = 2 * env.env_config["timestep"]
        self.default_foot_solimp = self.env.initial_mj_model.geom_solimp[self.env.foot_geom_indices]
        self.default_impratio = self.env.initial_mj_model.opt.impratio
        self.default_gravity = self.env.initial_mj_model.opt.gravity[2]
        self.default_density = self.env.initial_mj_model.opt.density
        self.default_viscosity = self.env.initial_mj_model.opt.viscosity

    def sample(self):
        sampled_friction_tangential = self.default_friction_tangential * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.friction_tangential_factor, high=self.friction_tangential_factor, size=self.default_friction_tangential.size))
        sampled_friction_torsional = self.default_friction_torsional * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.friction_torsional_factor, high=self.friction_torsional_factor, size=self.default_friction_torsional.size))
        sampled_friction_rolling = self.default_friction_rolling * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.friction_rolling_factor, high=self.friction_rolling_factor, size=self.default_friction_rolling.size))

        sampled_timeconst = np.maximum(self.default_timeconst * np.exp(self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.timeconst_log_range, high=self.timeconst_log_range, size=self.default_timeconst.size)), self.min_timeconst)
        sampled_dampratio = self.default_dampratio * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.dampratio_factor, high=self.dampratio_factor, size=self.default_dampratio.size))
        sampled_solimp_dmin = np.clip(self.default_foot_solimp[:, 0] * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.foot_solimp_factor, high=self.foot_solimp_factor, size=self.default_foot_solimp[:, 0].size)), 0.1, self.default_foot_solimp[:, 1])

        opt_impratio = np.maximum(self.default_impratio + (self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.add_impratio, high=self.add_impratio)), 1.0)

        sampled_z_gravity = self.default_gravity * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.z_gravity_factor, high=self.z_gravity_factor))
        sampled_xy_gravity = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.xy_gravity, high=self.xy_gravity, size=(2,))

        opt_density = self.default_density * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.density_factor, high=self.density_factor))

        opt_viscosity = self.default_viscosity * (1 + self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.viscosity_factor, high=self.viscosity_factor))

        self.env.internal_state["mj_model"].geom_friction[self.env.foot_geom_indices, 0] = sampled_friction_tangential
        self.env.internal_state["mj_model"].geom_friction[self.env.foot_geom_indices, 1] = sampled_friction_torsional
        self.env.internal_state["mj_model"].geom_friction[self.env.foot_geom_indices, 2] = sampled_friction_rolling
        self.env.internal_state["mj_model"].geom_solref[self.env.foot_geom_indices, 0] = sampled_timeconst
        self.env.internal_state["mj_model"].geom_solref[self.env.foot_geom_indices, 1] = sampled_dampratio
        self.env.internal_state["mj_model"].geom_solimp[self.env.foot_geom_indices, 0] = sampled_solimp_dmin
        self.env.internal_state["mj_model"].opt.impratio = opt_impratio
        self.env.internal_state["mj_model"].opt.gravity[0] = sampled_xy_gravity[0]
        self.env.internal_state["mj_model"].opt.gravity[1] = sampled_xy_gravity[1]
        self.env.internal_state["mj_model"].opt.gravity[2] = sampled_z_gravity
        self.env.internal_state["mj_model"].opt.density = opt_density
        self.env.internal_state["mj_model"].opt.viscosity = opt_viscosity
