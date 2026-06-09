import jax.numpy as jnp


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.ground_height = self.env.c_data.geom_xpos[self.env.floor_geom_id, 2]


    def init(self, internal_state):
        internal_state["center_height"] = jnp.full(self.env.nr_envs, self.ground_height)
        internal_state["robot_imu_height_over_ground"] = jnp.full(self.env.nr_envs, self.env.initial_imu_height - self.ground_height)


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
            in_contact = jnp.sum(global_corners[:, :, :, 2] > self.ground_height, axis=2)
        return in_contact


    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        center_height = internal_state["center_height"].reshape((self.env.nr_envs,) + (1,) * (x_in_m.ndim - 1))
        return jnp.ones_like(x_in_m) * center_height


    def pre_step(self, data, internal_state):
        internal_state["robot_imu_height_over_ground"] = data.site_xpos[:, self.env.imu_site_id, 2] - internal_state["center_height"]


    def post_step(self, data, mjx_model, internal_state, key):
        return data


    def sample(self, mjx_model, internal_state, key):
        return mjx_model
