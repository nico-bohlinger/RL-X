import numpy as np


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

        hfield_size = self.env.initial_mj_model.hfield_size[0]
        if hfield_size[0] != hfield_size[1]:
            raise ValueError("The heightfield is not square.")

        self.hfield_length = self.env.initial_mj_model.hfield_ncol[0]
        self.hfield_half_length_in_meters = hfield_size[0]
        self.max_possible_height = hfield_size[2]

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height


    def init(self):
        self.env.internal_state["center_height"] = 0.0
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.initial_imu_height - self.env.internal_state["center_height"]
        self.env.internal_state["current_height_field_data"] = self.env.initial_mj_model.hfield_data.reshape((self.hfield_length, self.hfield_length))


    def check_feet_floor_contact(self):
        contact_geom_pairs = self.env.internal_state["data"].contact.geom
        possible_contact_pairs = np.stack([np.full_like(self.env.foot_geom_indices, self.env.floor_geom_id), self.env.foot_geom_indices], axis=1)
        in_contact = np.any(np.all(contact_geom_pairs == possible_contact_pairs[:, None, :], axis=2), axis=1)

        return in_contact


    def check_flat_feet_floor_missing_contacts(self):
        if self.env.foot_type == "sphere":
            return np.zeros(self.env.nr_feet)
        elif self.env.foot_type == "box":
            feet_xpos = self.env.internal_state["data"].geom_xpos[self.env.foot_geom_indices]
            feet_xmat = self.env.internal_state["data"].geom_xmat[self.env.foot_geom_indices].reshape(-1, 3, 3)
            feet_sizes = self.env.internal_state["mj_model"].geom_size[self.env.foot_geom_indices]
            lower_base_corners = np.array([
                [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]
            ])
            corners = lower_base_corners[None, :, :] * feet_sizes[:, None, :]
            global_corners = np.einsum("fij,fgj->fgi", feet_xmat, corners) + feet_xpos[:, None, :]
            floor_height_at_corners = self.ground_height_at(global_corners[:, :, 0], global_corners[:, :, 1])
            in_contact = np.sum(global_corners[:, :, 2] > floor_height_at_corners, axis=1)
        return in_contact


    def ground_height_at(self, x_in_m, y_in_m):
        x = np.clip(np.round(x_in_m * self.one_meter_length + self.hfield_half_length).astype(np.int32), 0, self.hfield_length-1)
        y = np.clip(np.round(y_in_m * self.one_meter_length + self.hfield_half_length).astype(np.int32), 0, self.hfield_length-1)
        return self.env.internal_state["current_height_field_data"][y, x] * self.mujoco_height_scaling


    def pre_step(self):
        self.env.internal_state["robot_imu_height_over_ground"] = self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 2] - self.ground_height_at(self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 0], self.env.internal_state["data"].site_xpos[self.env.imu_site_id, 1])
    

    def post_step(self):
        min_edge = self.hfield_half_length_in_meters - 0.5
        max_edge = self.hfield_half_length_in_meters
        reached_edge = np.array(((min_edge < np.abs(self.env.internal_state["data"].qpos[0])) & (np.abs(self.env.internal_state["data"].qpos[0]) < max_edge)) | ((min_edge < np.abs(self.env.internal_state["data"].qpos[1])) & (np.abs(self.env.internal_state["data"].qpos[1]) < max_edge)))
        if reached_edge:
            qpos, qvel = self.env.initial_state_function.setup()
            self.env.internal_state["data"].qpos = qpos
            self.env.internal_state["data"].qvel = qvel
    

    def sample(self):
        isaac_height_field = self.diverse_terrain(
            wave_fn_min=self.wave_fn_min,
            wave_fn_max=self.wave_fn_max,
            wave_height=self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=0, high=self.env.internal_state["robot_dimensions_mean"] * self.wave_height_max_per_m_factor),
            random_height=self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=0, high=self.env.internal_state["robot_dimensions_mean"] * self.random_height_max_per_m_factor),
            block_probability=self.block_probability,
            block_length_in_meters=self.block_length_in_meters,
            block_height=self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=0, high=self.env.internal_state["robot_dimensions_mean"] * self.block_height_max_per_m_factor),
        )
        new_height_field_data = self.isaac_hf_to_mujoco_hf(isaac_height_field)

        self.env.internal_state["mj_model"].hfield_data = new_height_field_data

        self.env.internal_state["center_height"] = new_height_field_data[self.hfield_half_length * self.hfield_length + self.hfield_half_length] * self.mujoco_height_scaling
        self.env.internal_state["current_height_field_data"] = new_height_field_data.reshape(self.hfield_length, self.hfield_length)
    

    def isaac_hf_to_mujoco_hf(self, isaac_hf):
        hf = isaac_hf + np.abs(np.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)


    def diverse_terrain(self,
            wave_fn_min, wave_fn_max, wave_height,
            random_height,
            block_probability, block_height, block_length_in_meters
        ):
        I, J = np.meshgrid(np.arange(self.hfield_length),
                            np.arange(self.hfield_length),
                            indexing='ij')
        wave1 = np.sin(2 * np.pi * self.env.np_rng.uniform(low=wave_fn_min, high=wave_fn_max) * I / self.hfield_length)
        wave2 = np.sin(2 * np.pi * self.env.np_rng.uniform(low=wave_fn_min, high=wave_fn_max) * J / self.hfield_length)
        height_field_raw = wave_height * (wave1 + wave2)

        height_field_raw += self.env.np_rng.uniform(size=(self.hfield_length, self.hfield_length), low=-random_height, high=random_height)

        block_length = int(block_length_in_meters * self.one_meter_length)
        block_repeat = int(self.hfield_length * block_length)
        layer1 = np.repeat(self.env.np_rng.binomial(n=1, p=block_probability, size=((self.hfield_length * self.hfield_length) // block_repeat,)), block_repeat).reshape(self.hfield_length, self.hfield_length)
        layer2 = np.repeat(self.env.np_rng.binomial(n=1, p=block_probability, size=((self.hfield_length * self.hfield_length) // block_repeat,)), block_repeat).reshape(self.hfield_length, self.hfield_length).T
        height_field_raw += layer1.astype(float) * block_height + layer2.astype(float) * block_height

        return height_field_raw
