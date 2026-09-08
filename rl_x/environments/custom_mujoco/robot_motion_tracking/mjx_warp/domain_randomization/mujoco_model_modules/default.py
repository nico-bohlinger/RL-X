import numpy as np
import jax
import jax.numpy as jnp


class DefaultMujocoModel:
    def __init__(self, env):
        self.env = env

        self.enable_randomization = env.env_config["domain_randomization"]["mujoco_model"]["default"]["enable_randomization"]
        self.material_buckets = env.env_config["domain_randomization"]["mujoco_model"]["default"]["material_buckets"]
        self.robot_sliding_friction_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["robot_sliding_friction_range"]
        self.torso_com_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["torso_com_range"]
        self.object_sliding_friction_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_sliding_friction_range"]
        self.object_mass_add_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_mass_add_range"]
        self.object_inertia_x_scale_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_inertia_x_scale_range"]
        descendants = np.zeros((env.initial_mj_model.nbody, env.initial_mj_model.nbody), dtype=np.float32)
        for body in range(env.initial_mj_model.nbody):
            parent = body
            descendants[parent, body] = 1.0
            while parent != 0:
                parent = env.initial_mj_model.body_parentid[parent]
                descendants[parent, body] = 1.0
        self.descendants = jnp.asarray(descendants)
        self.model_axes = jax.tree.map(lambda _: None, env.initial_mjx_model).replace(geom_friction=0, body_ipos=0, body_mass=0, body_inertia=0, body_subtreemass=0)


    def sample(self, key, eval_mode):
        keys = jax.random.split(key, 6)
        enabled = self.enable_randomization & ~eval_mode
        friction = jnp.broadcast_to(self.env.initial_mjx_model.geom_friction, (self.env.nr_envs,) + self.env.initial_mjx_model.geom_friction.shape)
        buckets = jax.random.uniform(keys[0], (self.material_buckets,), minval=self.robot_sliding_friction_range[0], maxval=self.robot_sliding_friction_range[1])
        bucket_ids = jax.random.randint(keys[5], (self.env.nr_envs, len(self.env.robot_geom_ids)), 0, self.material_buckets)
        robot_friction = buckets[bucket_ids]
        friction = friction.at[:, self.env.robot_geom_ids, 0].set(jnp.where(enabled, robot_friction, friction[:, self.env.robot_geom_ids, 0]))

        com = jnp.broadcast_to(self.env.initial_mjx_model.body_ipos, (self.env.nr_envs,) + self.env.initial_mjx_model.body_ipos.shape)
        com = com.at[:, self.env.torso_body_id].add(jnp.where(enabled, jax.random.uniform(keys[1], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.torso_com_range), 0.0))

        mass = jnp.broadcast_to(self.env.initial_mjx_model.body_mass, (self.env.nr_envs,) + self.env.initial_mjx_model.body_mass.shape)
        inertia = jnp.broadcast_to(self.env.initial_mjx_model.body_inertia, (self.env.nr_envs,) + self.env.initial_mjx_model.body_inertia.shape)

        if self.env.has_object:
            object_friction = jax.random.uniform(keys[2], (self.env.nr_envs,), minval=self.object_sliding_friction_range[0], maxval=self.object_sliding_friction_range[1])
            friction = friction.at[:, self.env.object_geom_id, 0].set(jnp.where(enabled, object_friction, friction[:, self.env.object_geom_id, 0]))
            mass_offset = jnp.where(eval_mode, sum(self.object_mass_add_range) / 2, jax.random.uniform(keys[3], (self.env.nr_envs,), minval=self.object_mass_add_range[0], maxval=self.object_mass_add_range[1]))
            mass = mass.at[:, self.env.object_body_id].add(jnp.where(self.enable_randomization, mass_offset, 0.0))
            inertia = inertia.at[:, self.env.object_body_id].multiply(mass[:, self.env.object_body_id, None] / self.env.initial_mjx_model.body_mass[self.env.object_body_id])
            inertia = inertia.at[:, self.env.object_body_id, 0].multiply(jnp.where(enabled, jax.random.uniform(keys[4], (self.env.nr_envs,), minval=self.object_inertia_x_scale_range[0], maxval=self.object_inertia_x_scale_range[1]), 1.0))

        return self.env.initial_mjx_model.replace(geom_friction=friction, body_ipos=com, body_mass=mass, body_inertia=inertia, body_subtreemass=mass @ self.descendants.T)
