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
        trunk_velocity_key, trunk_add_key, joint_velocity_key, joint_position_key = jax.random.split(key, 4)

        max_trunk_velocity = jnp.minimum(jnp.sum(mjx_model.body_mass) * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)

        trunk_velocity_perturbation = internal_state["env_curriculum_coeff"] * jax.random.uniform(trunk_velocity_key, (6,), minval=-max_trunk_velocity, maxval=max_trunk_velocity)
        trunk_velocity_perturbation = jnp.where(jax.random.uniform(trunk_add_key) < self.trunk_velocity_add_chance, data.qvel[0:6] + trunk_velocity_perturbation, (trunk_velocity_perturbation * internal_state["env_curriculum_coeff"]) + (data.qvel[0:6] * (1 - internal_state["env_curriculum_coeff"])))
        trunk_velocity = jnp.where(should_randomize, trunk_velocity_perturbation, data.qvel[0:6])

        joint_velocity = data.qvel[6:] + internal_state["env_curriculum_coeff"] * jax.random.uniform(joint_velocity_key, (data.qvel[6:].shape[0],), minval=-self.max_joint_velocity, maxval=self.max_joint_velocity)
        joint_velocity = jnp.where(should_randomize, joint_velocity, data.qvel[6:])
        joint_position = data.qpos[7:] + internal_state["env_curriculum_coeff"] * jax.random.uniform(joint_position_key, (data.qpos[7:].shape[0],), minval=-self.max_joint_position, maxval=self.max_joint_position)
        joint_position = jnp.where(should_randomize, joint_position, data.qpos[7:])

        data = data.replace(
            qpos=jnp.concatenate([data.qpos[0:7], joint_position]),
            qvel=jnp.concatenate([trunk_velocity, joint_velocity])
        )
        
        return data
