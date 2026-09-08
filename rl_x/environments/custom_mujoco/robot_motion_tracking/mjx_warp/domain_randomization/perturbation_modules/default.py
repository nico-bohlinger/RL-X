import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class DefaultPerturbation:
    def __init__(self, env):
        self.env = env

        self.root_velocity = env.env_config["domain_randomization"]["perturbation"]["default"]["root_velocity"]
        self.enable_randomization = env.env_config["domain_randomization"]["perturbation"]["default"]["enable_randomization"]


    def apply(self, key, qvel, qpos, due, eval_mode):
        velocity = jax.random.uniform(key, (self.env.nr_envs, 6), minval=-1.0, maxval=1.0) * jnp.asarray(self.root_velocity)
        velocity = jnp.where((due & self.enable_randomization & ~eval_mode)[:, None], velocity, 0.0)

        qvel = qvel.at[:, self.env.root_linear_qvel_slice].add(velocity[:, :3])

        rotation = Rotation.from_quat(qpos[:, self.env.root_quaternion_qpos_slice][:, self.env.quaternion_wxyz_to_xyzw_indices])
        qvel = qvel.at[:, self.env.root_angular_qvel_slice].add(rotation.inv().apply(velocity[:, 3:]))
        return qvel
