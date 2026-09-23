import mujoco


class DefaultMujocoModel:
    def __init__(self, env):
        self.env = env

        self.enable_randomization = env.env_config["domain_randomization"]["mujoco_model"]["default"]["enable_randomization"]
        self.robot_sliding_friction_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["robot_sliding_friction_range"]
        self.material_buckets = env.env_config["domain_randomization"]["mujoco_model"]["default"]["material_buckets"]
        self.torso_com_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["torso_com_range"]
        self.object_sliding_friction_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_sliding_friction_range"]
        self.object_mass_add_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_mass_add_range"]
        self.object_inertia_x_scale_range = env.env_config["domain_randomization"]["mujoco_model"]["default"]["object_inertia_x_scale_range"]
        self.friction = env.internal_state["mj_model"].geom_friction.copy()
        self.com = env.internal_state["mj_model"].body_ipos.copy()
        self.mass = env.internal_state["mj_model"].body_mass.copy()
        self.inertia = env.internal_state["mj_model"].body_inertia.copy()


    def sample(self):
        self.env.internal_state["mj_model"].geom_friction[:] = self.friction
        self.env.internal_state["mj_model"].body_ipos[:] = self.com
        self.env.internal_state["mj_model"].body_mass[:] = self.mass
        self.env.internal_state["mj_model"].body_inertia[:] = self.inertia

        if self.enable_randomization and not self.env.internal_state["in_eval_mode"]:
            buckets = self.env.np_rng.uniform(*self.robot_sliding_friction_range, self.material_buckets)
            bucket_ids = self.env.np_rng.integers(0, self.material_buckets, len(self.env.robot_geom_ids))
            self.env.internal_state["mj_model"].geom_friction[self.env.robot_geom_ids, 0] = buckets[bucket_ids]
            self.env.internal_state["mj_model"].body_ipos[self.env.torso_body_id] += self.env.np_rng.uniform(-1.0, 1.0, 3) * self.torso_com_range

            if self.env.has_object:
                self.env.internal_state["mj_model"].geom_friction[self.env.object_geom_id, 0] = self.env.np_rng.uniform(*self.object_sliding_friction_range)
                self.env.internal_state["mj_model"].body_mass[self.env.object_body_id] += self.env.np_rng.uniform(*self.object_mass_add_range)
                self.env.internal_state["mj_model"].body_inertia[self.env.object_body_id] *= self.env.internal_state["mj_model"].body_mass[self.env.object_body_id] / self.mass[self.env.object_body_id]
                self.env.internal_state["mj_model"].body_inertia[self.env.object_body_id, 0] *= self.env.np_rng.uniform(*self.object_inertia_x_scale_range)

        elif self.enable_randomization and self.env.has_object:
            self.env.internal_state["mj_model"].body_mass[self.env.object_body_id] += sum(self.object_mass_add_range) / 2
            self.env.internal_state["mj_model"].body_inertia[self.env.object_body_id] *= self.env.internal_state["mj_model"].body_mass[self.env.object_body_id] / self.mass[self.env.object_body_id]

        mujoco.mj_setConst(self.env.internal_state["mj_model"], self.env.internal_state["data"])
