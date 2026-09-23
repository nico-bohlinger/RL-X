from scipy.spatial.transform import Rotation


class DefaultPerturbation:
    def __init__(self, env):
        self.env = env

        self.enable_randomization = env.env_config["domain_randomization"]["perturbation"]["default"]["enable_randomization"]
        self.root_velocity = env.env_config["domain_randomization"]["perturbation"]["default"]["root_velocity"]


    def apply(self, qvel, qpos, due):
        if due and self.enable_randomization and not self.env.internal_state["in_eval_mode"]:
            velocity = self.env.np_rng.uniform(-1.0, 1.0, 6) * self.root_velocity
            qvel[self.env.root_linear_qvel_slice] += velocity[:3]

            rotation = Rotation.from_quat(qpos[self.env.root_quaternion_qpos_slice][self.env.quaternion_wxyz_to_xyzw_indices])
            qvel[self.env.root_angular_qvel_slice] += rotation.inv().apply(velocity[3:])

        return qvel
