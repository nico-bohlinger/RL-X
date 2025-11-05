import mujoco
import jax.numpy as jnp


class HeightSamplesExteroceptiveObservation:
    def __init__(self, env,
                 measured_points_x=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 measured_points_y=[-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
                ):
        self.env = env
        self.measured_points_x = measured_points_x
        self.measured_points_y = measured_points_y

        self.nr_exteroceptive_observations = len(self.measured_points_x) * len(self.measured_points_y)

        self.floor_is_plane = self.env.initial_mj_model.geom_type[self.env.floor_geom_id] == mujoco.mjtGeom.mjGEOM_PLANE
        self.floor_is_hfield = self.env.initial_mj_model.geom_type[self.env.floor_geom_id] == mujoco.mjtGeom.mjGEOM_HFIELD

        if self.floor_is_hfield:
            hfield_size = self.env.initial_mjx_model.hfield_size[0]
            self.hfield_length = self.env.initial_mjx_model.hfield_ncol[0]
            self.hfield_half_length = self.hfield_length // 2
            self.hfield_half_length_in_meters = hfield_size[0]
            self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
            self.max_possible_height = hfield_size[2]
            self.mujoco_height_scaling = self.max_possible_height

            grid_x, grid_y = jnp.meshgrid(jnp.array(self.measured_points_x), jnp.array(self.measured_points_y), indexing='ij')
            self.base_height_points = jnp.stack([grid_x.flatten(), grid_y.flatten()], axis=1)


    def get_exteroceptive_observation(self, data, mjx_model, internal_state):
        if self.floor_is_hfield:
            rotation_angle = internal_state["imu_orientation_euler"][2]
            global_height_points_0 = self.base_height_points[:, 0] * jnp.cos(rotation_angle) - self.base_height_points[:, 1] * jnp.sin(rotation_angle)
            global_height_points_1 = self.base_height_points[:, 0] * jnp.sin(rotation_angle) + self.base_height_points[:, 1] * jnp.cos(rotation_angle)
            global_height_points = jnp.stack([global_height_points_0, global_height_points_1], axis=1)
            global_height_points += data.qpos[:2]

            global_height_points = (global_height_points * self.one_meter_length).astype(jnp.int32)
            px = global_height_points[:, 0].reshape(-1)
            py = global_height_points[:, 1].reshape(-1)
            px += self.hfield_half_length
            py += self.hfield_half_length
            px = jnp.clip(px, 0, self.hfield_length-2)
            py = jnp.clip(py, 0, self.hfield_length-2)
            
            current_height_field_data = internal_state["current_height_field_data"]
            heights1 = current_height_field_data[py, px]
            heights2 = current_height_field_data[py+1, px]
            heights3 = current_height_field_data[py, px+1]
            heights = jnp.minimum(heights1, heights2)
            heights = jnp.minimum(heights, heights3)

            sampled_heights = data.qpos[2] - heights * self.mujoco_height_scaling
            return jnp.clip(sampled_heights, -self.max_possible_height, self.max_possible_height)
        
        elif self.floor_is_plane:
            return internal_state["robot_imu_height_over_ground"] * jnp.ones(self.nr_exteroceptive_observations, dtype=jnp.float32)
