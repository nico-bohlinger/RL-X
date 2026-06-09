import jax
import jax.numpy as jnp


class HFieldDiverseTerrainGeneration:
    def __init__(self, env):
        self.env = env

        self.wave_fn_min = self.env.env_config["terrain"]["wave_fn_min"]
        self.wave_fn_max = self.env.env_config["terrain"]["wave_fn_max"]
        self.wave_height_max_per_m_factor = self.env.env_config["terrain"]["wave_height_max_per_m_factor"]
        self.random_height_max_per_m_factor = self.env.env_config["terrain"]["random_height_max_per_m_factor"]
        self.block_probability = self.env.env_config["terrain"]["block_probability"]
        self.block_length_in_meters = self.env.env_config["terrain"]["block_length_in_meters"]
        self.block_height_max_per_m_factor = self.env.env_config["terrain"]["block_height_max_per_m_factor"]

        hfield_size = self.env.initial_mjx_model.hfield_size[0]
        if hfield_size[0] != hfield_size[1]:
            raise ValueError("The heightfield is not square.")

        self.hfield_length = int(self.env.initial_mjx_model.hfield_ncol[0])
        self.hfield_half_length_in_meters = float(hfield_size[0])
        self.max_possible_height = float(hfield_size[2])

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height

        # mjx-warp has no per-world heightfield, so the model is built with one hfield copy per
        # environment (see environment.py) and the patched collision kernel routes world i to copy i
        # via geom_dataid[g1] + worldid. mjx_model.hfield_data is the flat (nr_envs * nr_cells,) buffer.
        # The per-env current heightfield lives in internal_state so it is merged per environment on
        # reset (instance attributes cannot hold per-episode state under jit).
        self.block_length = int(self.block_length_in_meters * self.one_meter_length)
        self.block_repeat = int(self.hfield_length * self.block_length)
        self.nr_blocks = (self.hfield_length * self.hfield_length) // self.block_repeat

        grid_i, grid_j = jnp.meshgrid(jnp.arange(self.hfield_length), jnp.arange(self.hfield_length), indexing="ij")
        self.grid_i = grid_i
        self.grid_j = grid_j


    def init(self, internal_state):
        nr_envs = self.env.nr_envs
        internal_state["current_height_field_data"] = jnp.zeros((nr_envs, self.hfield_length, self.hfield_length))
        internal_state["center_height"] = jnp.zeros(nr_envs)
        internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - internal_state["center_height"]


    def check_feet_floor_contact(self, data):
        nr_envs = self.env.nr_envs
        contact_geom = data._impl.contact__geom
        contact_dist = data._impl.contact__dist
        contact_worldid = data._impl.contact__worldid
        nr_contacts = contact_geom.shape[0]
        valid_contact = jnp.arange(nr_contacts) < data._impl.nacon[0]

        contact_pairs = jnp.stack([jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        contact_pairs_rev = jnp.stack([self.env.foot_geom_indices, jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id)], axis=1)
        mask1 = (contact_geom[:, None, :] == contact_pairs[None, :, :]).all(axis=2)
        mask2 = (contact_geom[:, None, :] == contact_pairs_rev[None, :, :]).all(axis=2)
        foot_contact_mask = mask1 | mask2
        penetrating = (contact_dist < 0.0) & valid_contact
        in_contact_per_contact = (foot_contact_mask & penetrating[:, None]).astype(jnp.float32)

        worldid_clamped = jnp.clip(contact_worldid, 0, nr_envs - 1)
        in_contact = jnp.zeros((nr_envs, self.env.nr_feet), dtype=jnp.float32)
        in_contact = in_contact.at[worldid_clamped].add(in_contact_per_contact)

        return in_contact > 0.0


    def check_flat_feet_floor_missing_contacts(self, data, mjx_model, internal_state):
        nr_envs = self.env.nr_envs
        if self.env.foot_type == "sphere":
            return jnp.zeros((nr_envs, self.env.nr_feet))
        elif self.env.foot_type == "box":
            geom_size = mjx_model.geom_size
            if geom_size.ndim == 2:
                geom_size = jnp.broadcast_to(geom_size, (nr_envs,) + geom_size.shape)
            feet_xpos = data.geom_xpos[:, self.env.foot_geom_indices]
            feet_xmat = data.geom_xmat[:, self.env.foot_geom_indices].reshape(nr_envs, self.env.nr_feet, 3, 3)
            feet_sizes = geom_size[:, self.env.foot_geom_indices]
            lower_base_corners = jnp.array([
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
            ])
            corners = lower_base_corners[None, None, :, :] * feet_sizes[:, :, None, :]
            global_corners = jnp.einsum("nfij,nfgj->nfgi", feet_xmat, corners) + feet_xpos[:, :, None, :]
            floor_height_at_corners = self.ground_height_at(internal_state, global_corners[:, :, :, 0], global_corners[:, :, :, 1])
            in_contact = jnp.sum(global_corners[:, :, :, 2] > floor_height_at_corners, axis=2)
        return in_contact


    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        x = jnp.clip(jnp.round(x_in_m * self.one_meter_length + self.hfield_half_length).astype(jnp.int32), 0, self.hfield_length - 1)
        y = jnp.clip(jnp.round(y_in_m * self.one_meter_length + self.hfield_half_length).astype(jnp.int32), 0, self.hfield_length - 1)
        env_idx = jnp.arange(self.env.nr_envs).reshape((self.env.nr_envs,) + (1,) * (x.ndim - 1))
        return internal_state["current_height_field_data"][env_idx, y, x] * self.mujoco_height_scaling


    def pre_step(self, data, internal_state):
        imu_xpos = data.site_xpos[:, self.env.imu_site_id]
        internal_state["robot_imu_height_over_ground"] = imu_xpos[:, 2] - self.ground_height_at(internal_state, imu_xpos[:, 0], imu_xpos[:, 1])


    def post_step(self, data, mjx_model, internal_state, key):
        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        reached_edge = ((min_edge < jnp.abs(data.qpos[:, 0])) & (jnp.abs(data.qpos[:, 0]) < max_edge)) | ((min_edge < jnp.abs(data.qpos[:, 1])) & (jnp.abs(data.qpos[:, 1]) < max_edge))
        qpos, qvel = self.env.initial_state_function.setup(mjx_model, internal_state, key)
        new_data = data.replace(
            qpos=jnp.where(reached_edge[:, None], qpos, data.qpos),
            qvel=jnp.where(reached_edge[:, None], qvel, data.qvel),
        )
        return new_data


    def sample(self, mjx_model, internal_state, key):
        nr_envs = self.env.nr_envs
        keys = jax.random.split(key, 4)
        curriculum_coeff = internal_state["env_curriculum_coeff"]
        robot_dimensions_mean = internal_state["robot_dimensions_mean"]

        wave_height = curriculum_coeff * jax.random.uniform(keys[0], (nr_envs,)) * (robot_dimensions_mean * self.wave_height_max_per_m_factor)
        random_height = curriculum_coeff * jax.random.uniform(keys[1], (nr_envs,)) * (robot_dimensions_mean * self.random_height_max_per_m_factor)
        block_height = curriculum_coeff * jax.random.uniform(keys[2], (nr_envs,)) * (robot_dimensions_mean * self.block_height_max_per_m_factor)

        new_field = self.diverse_terrain(wave_height, random_height, block_height, keys[3])
        new_field = new_field + jnp.abs(jnp.min(new_field, axis=(1, 2), keepdims=True))
        new_field = new_field / self.mujoco_height_scaling

        internal_state["current_height_field_data"] = new_field
        internal_state["center_height"] = new_field[:, self.hfield_half_length, self.hfield_half_length] * self.mujoco_height_scaling

        return mjx_model.replace(hfield_data=new_field.reshape(-1))


    def diverse_terrain(self, wave_height, random_height, block_height, key):
        nr_envs = self.env.nr_envs
        length = self.hfield_length
        keys = jax.random.split(key, 5)

        wave_fn_1 = jax.random.uniform(keys[0], (nr_envs,), minval=self.wave_fn_min, maxval=self.wave_fn_max)
        wave_fn_2 = jax.random.uniform(keys[1], (nr_envs,), minval=self.wave_fn_min, maxval=self.wave_fn_max)
        wave1 = jnp.sin(2 * jnp.pi * wave_fn_1[:, None, None] * self.grid_i[None] / length)
        wave2 = jnp.sin(2 * jnp.pi * wave_fn_2[:, None, None] * self.grid_j[None] / length)
        height_field = wave_height[:, None, None] * (wave1 + wave2)

        height_field = height_field + jax.random.uniform(keys[2], (nr_envs, length, length), minval=-1.0, maxval=1.0) * random_height[:, None, None]

        layer1 = jnp.repeat(jax.random.bernoulli(keys[3], self.block_probability, (nr_envs, self.nr_blocks)), self.block_repeat, axis=1).reshape(nr_envs, length, length)
        layer2 = jnp.repeat(jax.random.bernoulli(keys[4], self.block_probability, (nr_envs, self.nr_blocks)), self.block_repeat, axis=1).reshape(nr_envs, length, length).transpose(0, 2, 1)
        height_field = height_field + layer1.astype(float) * block_height[:, None, None] + layer2.astype(float) * block_height[:, None, None]

        return height_field
