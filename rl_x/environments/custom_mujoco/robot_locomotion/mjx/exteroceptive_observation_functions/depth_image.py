import mujoco
import jax
import jax.numpy as jnp


class DepthImageExteroceptiveObservation:
    def __init__(self, env, fov_y=45, p_x=16, p_y=16, f=0.1, min_depth=0.28, max_depth=10.0):
        self.env = env
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.nr_exteroceptive_observations = p_x * p_y

        floor_geom_group = self.env.initial_mjx_model.geom_group[self.env.floor_geom_id]
        if any(self.env.initial_mjx_model.geom_group[:self.env.floor_geom_id] == floor_geom_group) or any(self.env.initial_mjx_model.geom_group[self.env.floor_geom_id + 1:] == floor_geom_group):
            raise ValueError("The geom group of the floor geom is not unique.")
        
        self.floor_is_plane = self.env.initial_mj_model.geom_type[self.env.floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
        self.floor_is_hfield = self.env.initial_mj_model.geom_type[self.env.floor_geom_id] == mujoco.mjtGeom.mjGEOM_HFIELD

        if self.floor_is_plane:
            self.floor_size = self.env.initial_mjx_model.geom_size[self.env.floor_geom_id]
            self.ray_geom_mask = [False] * mujoco.mjNGROUP
            self.ray_geom_mask[floor_geom_group] = True
            self.ray_geom_mask = tuple(self.ray_geom_mask)
        elif self.floor_is_hfield:
            self.hfield_nrow = self.env.initial_mjx_model.hfield_nrow[0]
            self.hfield_ncol = self.env.initial_mjx_model.hfield_ncol[0]
            self.hfield_size = self.env.initial_mjx_model.hfield_size[0]
            self.x_min = -self.hfield_size[0]
            self.x_max = self.hfield_size[0]
            self.y_min = -self.hfield_size[1]
            self.y_max = self.hfield_size[1]
            self.x_range = self.x_max - self.x_min
            self.y_range = self.y_max - self.y_min
            self.pixel_per_meter = self.hfield_ncol / self.x_range
            self.num_visible_pixels = int((max_depth - min_depth) * self.pixel_per_meter)
            self.num_pixels_on_hfield = self.hfield_ncol
            self.num_samples = min(self.num_visible_pixels, self.num_pixels_on_hfield)
            self.t_vals = jnp.linspace(min_depth, max_depth, self.num_samples)
            self.t_vals = self.t_vals[None, :, None]
        
        _fov_y = jnp.deg2rad(fov_y)
        h_ip = jnp.tan(_fov_y / 2) * 2 * f
        w_ip = h_ip * (p_x / p_y)  # Square pixels
        delta = w_ip / (2 * p_x)
        x_coords_ip = jnp.linspace(-w_ip / 2 + delta, w_ip / 2 - delta, p_x)
        y_coords_ip = jnp.flip(jnp.linspace(-h_ip / 2 + delta, h_ip / 2 - delta, p_y))
        xx, yy = jnp.meshgrid(x_coords_ip, y_coords_ip)
        self.vecs_cf = jnp.concatenate([jnp.expand_dims(xx, axis=2), jnp.expand_dims(yy, axis=2), -1 * jnp.ones(xx.shape + (1,)) * f], axis=2)

        self.camera_id = mujoco.mj_name2id(self.env.initial_mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_camera")
    

    def get_exteroceptive_observation(self, data, mjx_model, internal_state):
        ray_origin = data.cam_xpos[self.camera_id]
        cam_bases = data.cam_xmat[self.camera_id]
        vecs_gf = jnp.matmul(cam_bases, jnp.expand_dims(self.vecs_cf, 3))
        vecs_gf = jnp.squeeze(vecs_gf)
        def normalise(v):
            return v / jnp.linalg.norm(v)
        b_normalise = jax.vmap(jax.vmap(normalise))
        vecs_gf = b_normalise(vecs_gf)
        ray_directions = vecs_gf.reshape((-1, 3))

        if self.floor_is_plane:
            distances = -ray_origin[2] / ray_directions[:, 2]
            valid = ray_directions[:, 2] <= -mujoco.mjMINVAL
            valid &= distances >= 0
            p = ray_origin[0:2] + jnp.expand_dims(distances, -1) * ray_directions[:, 0:2]
            valid &= jnp.all((self.floor_size[0:2] <= 0) | (jnp.abs(p) <= self.floor_size[0:2]), axis=1)
            distances = jnp.where(valid, distances, self.max_depth)
            distances = jnp.clip(distances, self.min_depth, self.max_depth)
        
        elif self.floor_is_hfield:
            dir = ray_directions[:, None, :]
            positions = ray_origin + self.t_vals * dir
            x = positions[:, :, 0]
            y = positions[:, :, 1]
            z_ray = positions[:, :, 2]

            x_clipped = jnp.clip(x, self.x_min, self.x_max)
            y_clipped = jnp.clip(y, self.y_min, self.y_max)
            u = (x_clipped - self.x_min) / self.x_range
            v = (y_clipped - self.y_min) / self.y_range
            i = u * (self.hfield_ncol - 1)
            j = v * (self.hfield_nrow - 1)
            i0 = jnp.floor(i).astype(int)
            j0 = jnp.floor(j).astype(int)
            h_vals = mjx_model.hfield_data.reshape((self.hfield_nrow, self.hfield_ncol))[j0, i0] * self.hfield_size[2]

            below = z_ray <= h_vals

            def find_first_true(arr):
                idx = jnp.argmax(arr)
                is_found = arr[idx]
                idx = jnp.where(is_found, idx, arr.shape[0])
                return idx
            
            first_idx = jax.vmap(find_first_true)(below)

            distances = self.min_depth + (self.max_depth - self.min_depth) * first_idx / (self.num_samples - 1)
            distances = jnp.where(first_idx < self.num_samples, distances, self.max_depth)

        return distances
