import jax
import jax.numpy as jnp


class DefaultDRMuJoCoModel:
    def __init__(self, env):
        self.env = env

        self.friction_tangential_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_tangential_factor"]
        self.friction_torsional_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_torsional_factor"]
        self.friction_rolling_factor = env.env_config["domain_randomization"]["mujoco_model"]["friction_rolling_factor"]
        self.stiffness_factor = env.env_config["domain_randomization"]["mujoco_model"]["stiffness_factor"]
        self.damping_factor = env.env_config["domain_randomization"]["mujoco_model"]["damping_factor"]
        self.foot_solimp_factor = env.env_config["domain_randomization"]["mujoco_model"]["foot_solimp_factor"]
        self.add_impratio = env.env_config["domain_randomization"]["mujoco_model"]["add_impratio"]
        self.xy_gravity = env.env_config["domain_randomization"]["mujoco_model"]["xy_gravity"]
        self.z_gravity_factor = env.env_config["domain_randomization"]["mujoco_model"]["z_gravity_factor"]
        self.density_factor = env.env_config["domain_randomization"]["mujoco_model"]["density_factor"]
        self.viscosity_factor = env.env_config["domain_randomization"]["mujoco_model"]["viscosity_factor"]

        self.default_friction_tangential = self.env.initial_mjx_model.geom_friction[self.env.foot_geom_indices, 0]
        self.default_friction_torsional = self.env.initial_mjx_model.geom_friction[self.env.foot_geom_indices, 1]
        self.default_friction_rolling = self.env.initial_mjx_model.geom_friction[self.env.foot_geom_indices, 2]
        self.default_stiffness = self.env.initial_mjx_model.geom_solref[:, 0]
        self.default_damping = self.env.initial_mjx_model.geom_solref[:, 1]
        self.default_foot_solimp = self.env.initial_mjx_model.geom_solimp[self.env.foot_geom_indices]
        self.default_impratio = self.env.initial_mjx_model.opt.impratio
        self.default_gravity = self.env.initial_mjx_model.opt.gravity[2]
        self.default_density = self.env.initial_mjx_model.opt.density
        self.default_viscosity = self.env.initial_mjx_model.opt.viscosity

        if self.env.foot_type == "sphere":
            self.foot_size_height_index = 0
        elif self.env.foot_type == "box":
            self.foot_size_height_index = 2


    def sample(self, internal_state, mjx_model, should_randomize, key):
        keys = jax.random.split(key, 11)

        sampled_friction_tangential = self.default_friction_tangential * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[0], minval=-self.friction_tangential_factor, maxval=self.friction_tangential_factor, shape=self.default_friction_tangential.shape))
        sampled_friction_torsional = self.default_friction_torsional * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[1], minval=-self.friction_torsional_factor, maxval=self.friction_torsional_factor, shape=self.default_friction_torsional.shape))
        sampled_friction_rolling = self.default_friction_rolling * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[2], minval=-self.friction_rolling_factor, maxval=self.friction_rolling_factor, shape=self.default_friction_rolling.shape))
        geom_friction = mjx_model.geom_friction.at[self.env.foot_geom_indices].set(jnp.array([sampled_friction_tangential, sampled_friction_torsional, sampled_friction_rolling]).T)

        sampled_stiffness = self.default_stiffness * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[3], minval=-self.stiffness_factor, maxval=self.stiffness_factor))
        sampled_damping = self.default_damping * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[4], minval=-self.damping_factor, maxval=self.damping_factor))
        geom_solref = mjx_model.geom_solref.at[:, 0].set(sampled_stiffness)
        geom_solref = geom_solref.at[:, 1].set(sampled_damping)

        foot_solimp = self.default_foot_solimp.at[:, 2].set(mjx_model.geom_size[self.env.foot_geom_indices, self.foot_size_height_index])
        foot_solimp = jnp.clip(foot_solimp * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[5], minval=-self.foot_solimp_factor, maxval=self.foot_solimp_factor, shape=self.default_foot_solimp.shape)), jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 1.0, 1.0, 6.0]))
        geom_solimp = mjx_model.geom_solimp.at[self.env.foot_geom_indices].set(foot_solimp)

        opt_impratio = jnp.maximum(self.default_impratio + (internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[6], minval=-self.add_impratio, maxval=self.add_impratio)), 1.0)

        sampled_z_gravity = self.default_gravity * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[7], minval=-self.z_gravity_factor, maxval=self.z_gravity_factor))
        sampled_xy_gravity = internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[8], minval=-self.xy_gravity, maxval=self.xy_gravity, shape=(2,))
        opt_gravity = mjx_model.opt.gravity.at[:].set(jnp.array([sampled_xy_gravity[0], sampled_xy_gravity[1], sampled_z_gravity]))

        opt_density = self.default_density * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[9], minval=-self.density_factor, maxval=self.density_factor))

        opt_viscosity = self.default_viscosity * (1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[10], minval=-self.viscosity_factor, maxval=self.viscosity_factor))

        new_mjx_model = mjx_model.tree_replace(
            {
                "geom_friction": geom_friction,
                "geom_solref": geom_solref,
                "geom_solimp": geom_solimp,
                "opt.impratio": opt_impratio,
                "opt.gravity": opt_gravity,
                "opt.density": opt_density,
                "opt.viscosity": opt_viscosity
            }
        )
        mjx_model = jax.lax.cond(should_randomize, lambda x: new_mjx_model, lambda x: mjx_model, None)

        return mjx_model
