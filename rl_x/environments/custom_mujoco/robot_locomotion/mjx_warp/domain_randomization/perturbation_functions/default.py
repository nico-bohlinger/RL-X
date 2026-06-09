import jax
import jax.numpy as jnp


class DefaultDRPerturbation:
    def __init__(self, env):
        self.env = env

        self.trunk_velocity_clip_mass_factor = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_clip_mass_factor"]
        self.trunk_velocity_clip_limit = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_clip_limit"]
        self.trunk_velocity_add_chance = env.env_config["domain_randomization"]["perturbation"]["trunk_velocity_add_chance"]
        self.max_joint_velocity = env.env_config["domain_randomization"]["perturbation"]["max_joint_velocity"]
        self.max_joint_position = env.env_config["domain_randomization"]["perturbation"]["max_joint_position"]


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        nr_envs = self.env.nr_envs
        trunk_velocity_key, trunk_add_key, joint_velocity_key, joint_position_key = jax.random.split(key, 4)
        curriculum_coeff = internal_state["env_curriculum_coeff"][:, None]

        total_mass = jnp.sum(mjx_model.body_mass, axis=-1)
        max_trunk_velocity = jnp.minimum(total_mass * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)
        if jnp.ndim(max_trunk_velocity) == 0:
            max_trunk_velocity = jnp.full(nr_envs, max_trunk_velocity)
        max_trunk_velocity = max_trunk_velocity[:, None]

        trunk_velocity_perturbation = curriculum_coeff * jax.random.uniform(trunk_velocity_key, (nr_envs, 6), minval=-max_trunk_velocity, maxval=max_trunk_velocity)
        trunk_velocity_perturbation = jnp.where(jax.random.uniform(trunk_add_key, (nr_envs, 1)) < self.trunk_velocity_add_chance, data.qvel[:, 0:6] + trunk_velocity_perturbation, (trunk_velocity_perturbation * curriculum_coeff) + (data.qvel[:, 0:6] * (1 - curriculum_coeff)))
        trunk_velocity = jnp.where(should_randomize[:, None], trunk_velocity_perturbation, data.qvel[:, 0:6])

        joint_velocity = data.qvel[:, 6:] + curriculum_coeff * jax.random.uniform(joint_velocity_key, data.qvel[:, 6:].shape, minval=-self.max_joint_velocity, maxval=self.max_joint_velocity)
        joint_velocity = jnp.where(should_randomize[:, None], joint_velocity, data.qvel[:, 6:])
        joint_position = data.qpos[:, 7:] + curriculum_coeff * jax.random.uniform(joint_position_key, data.qpos[:, 7:].shape, minval=-self.max_joint_position, maxval=self.max_joint_position)
        joint_position = jnp.where(should_randomize[:, None], joint_position, data.qpos[:, 7:])

        data = data.replace(
            qpos=jnp.concatenate([data.qpos[:, 0:7], joint_position], axis=-1),
            qvel=jnp.concatenate([trunk_velocity, joint_velocity], axis=-1)
        )

        return data
