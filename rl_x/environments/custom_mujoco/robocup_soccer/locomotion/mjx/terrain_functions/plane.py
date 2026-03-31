import mujoco
import jax.numpy as jnp


class PlaneTerrainGeneration:
    def __init__(self, env):
        self.env = env
        self.ground_height = self.env.c_data.geom_xpos[self.env.floor_geom_id, 2]


    def init(self, internal_state):
        internal_state["center_height"] = self.ground_height
        internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - internal_state["center_height"]
    

    def check_feet_floor_contact(self, data):
        contact_pairs = jnp.stack([jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        contact_pairs_rev = jnp.stack([self.env.foot_geom_indices, jnp.full_like(self.env.foot_geom_indices, self.env.floor_geom_id)], axis=1)
        mask1 = (data._impl.contact.geom[None, :, :] == contact_pairs[:, None, :]).all(axis=2)
        mask2 = (data._impl.contact.geom[None, :, :] == contact_pairs_rev[:, None, :]).all(axis=2)
        mask = mask1 | mask2
        masekd_dist = jnp.where(mask, data._impl.contact.dist[None, :], 1e4)
        indices = masekd_dist.argmin(axis=1)
        dists = data._impl.contact.dist[indices] * mask[jnp.arange(mask.shape[0]), indices]
        in_contact = dists < 0.0

        return in_contact
    

    def check_flat_feet_floor_missing_contacts(self, data, mjx_model, internal_state):
        if self.env.foot_type == "sphere":
            return jnp.zeros(self.env.nr_feet)
        elif self.env.foot_type == "box":
            feet_xpos = data.geom_xpos[self.env.foot_geom_indices]
            feet_xmat = data.geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
            feet_sizes = mjx_model.geom_size[self.env.foot_geom_indices]
            lower_base_corners = jnp.array([
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
            ])
            corners = lower_base_corners[None, :, :] * feet_sizes[:, None, :]
            global_corners = jnp.einsum("fij,fgj->fgi", feet_xmat, corners) + feet_xpos[:, None, :]
            in_contact = jnp.sum(global_corners[:, :, 2] > self.ground_height, axis=1)
        return in_contact


    def ground_height_at(self, internal_state, x_in_m, y_in_m):
        return jnp.ones_like(x_in_m) * internal_state["center_height"]


    def pre_step(self, data, internal_state):
        internal_state["robot_imu_height_over_ground"] = data.site_xpos[self.env.imu_site_id, 2] - internal_state["center_height"]


    def post_step(self, data, mjx_model, internal_state, key):
        return data


    def sample(self, mjx_model, internal_state, key):
        return mjx_model
