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
        nr_envs = self.env.nr_envs
        keys = jax.random.split(key, 11)
        curriculum_coeff = internal_state["env_curriculum_coeff"][:, None]

        geom_friction = mjx_model.geom_friction
        geom_solref = mjx_model.geom_solref
        geom_solimp = mjx_model.geom_solimp
        geom_size = mjx_model.geom_size
        if geom_friction.ndim == 2:
            geom_friction = jnp.broadcast_to(geom_friction, (nr_envs,) + geom_friction.shape)
        if geom_solref.ndim == 2:
            geom_solref = jnp.broadcast_to(geom_solref, (nr_envs,) + geom_solref.shape)
        if geom_solimp.ndim == 2:
            geom_solimp = jnp.broadcast_to(geom_solimp, (nr_envs,) + geom_solimp.shape)
        if geom_size.ndim == 2:
            geom_size = jnp.broadcast_to(geom_size, (nr_envs,) + geom_size.shape)

        sampled_friction_tangential = self.default_friction_tangential * (1 + curriculum_coeff * jax.random.uniform(keys[0], minval=-self.friction_tangential_factor, maxval=self.friction_tangential_factor, shape=(nr_envs,) + self.default_friction_tangential.shape))
        sampled_friction_torsional = self.default_friction_torsional * (1 + curriculum_coeff * jax.random.uniform(keys[1], minval=-self.friction_torsional_factor, maxval=self.friction_torsional_factor, shape=(nr_envs,) + self.default_friction_torsional.shape))
        sampled_friction_rolling = self.default_friction_rolling * (1 + curriculum_coeff * jax.random.uniform(keys[2], minval=-self.friction_rolling_factor, maxval=self.friction_rolling_factor, shape=(nr_envs,) + self.default_friction_rolling.shape))
        # mujoco_warp requires strictly positive friction coefficients, so a minimum friction floor is applied here (deviation from the mjx version which does not clip)
        sampled_friction = jnp.maximum(jnp.stack([sampled_friction_tangential, sampled_friction_torsional, sampled_friction_rolling], axis=-1), 1e-5)
        geom_friction = geom_friction.at[:, self.env.foot_geom_indices].set(sampled_friction)

        sampled_stiffness = self.default_stiffness * (1 + curriculum_coeff * jax.random.uniform(keys[3], (nr_envs, 1), minval=-self.stiffness_factor, maxval=self.stiffness_factor))
        sampled_damping = self.default_damping * (1 + curriculum_coeff * jax.random.uniform(keys[4], (nr_envs, 1), minval=-self.damping_factor, maxval=self.damping_factor))
        geom_solref = geom_solref.at[:, :, 0].set(sampled_stiffness)
        geom_solref = geom_solref.at[:, :, 1].set(sampled_damping)

        foot_solimp = jnp.broadcast_to(self.default_foot_solimp, (nr_envs,) + self.default_foot_solimp.shape)
        foot_solimp = foot_solimp.at[:, :, 2].set(geom_size[:, self.env.foot_geom_indices, self.foot_size_height_index])
        foot_solimp = jnp.clip(foot_solimp * (1 + curriculum_coeff[:, :, None] * jax.random.uniform(keys[5], minval=-self.foot_solimp_factor, maxval=self.foot_solimp_factor, shape=(nr_envs,) + self.default_foot_solimp.shape)), jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]), jnp.array([1.0, 1.0, 1.0, 1.0, 6.0]))
        geom_solimp = geom_solimp.at[:, self.env.foot_geom_indices].set(foot_solimp)

        opt_impratio = jnp.maximum(self.default_impratio + (curriculum_coeff[:, 0] * jax.random.uniform(keys[6], (nr_envs,), minval=-self.add_impratio, maxval=self.add_impratio)), 1.0)

        sampled_z_gravity = self.default_gravity * (1 + curriculum_coeff[:, 0] * jax.random.uniform(keys[7], (nr_envs,), minval=-self.z_gravity_factor, maxval=self.z_gravity_factor))
        sampled_xy_gravity = curriculum_coeff * jax.random.uniform(keys[8], (nr_envs, 2), minval=-self.xy_gravity, maxval=self.xy_gravity)
        opt_gravity = jnp.stack([sampled_xy_gravity[:, 0], sampled_xy_gravity[:, 1], sampled_z_gravity], axis=-1)

        opt_density = self.default_density * (1 + curriculum_coeff[:, 0] * jax.random.uniform(keys[9], (nr_envs,), minval=-self.density_factor, maxval=self.density_factor))

        opt_viscosity = self.default_viscosity * (1 + curriculum_coeff[:, 0] * jax.random.uniform(keys[10], (nr_envs,), minval=-self.viscosity_factor, maxval=self.viscosity_factor))

        geom_friction = jnp.where(should_randomize[:, None, None], geom_friction, mjx_model.geom_friction)
        geom_solref = jnp.where(should_randomize[:, None, None], geom_solref, mjx_model.geom_solref)
        geom_solimp = jnp.where(should_randomize[:, None, None], geom_solimp, mjx_model.geom_solimp)
        opt_impratio = jnp.where(should_randomize, opt_impratio, mjx_model.opt.impratio)
        opt_gravity = jnp.where(should_randomize[:, None], opt_gravity, mjx_model.opt.gravity)
        opt_density = jnp.where(should_randomize, opt_density, mjx_model.opt.density)
        opt_viscosity = jnp.where(should_randomize, opt_viscosity, mjx_model.opt.viscosity)

        mjx_model = mjx_model.tree_replace(
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

        return mjx_model
