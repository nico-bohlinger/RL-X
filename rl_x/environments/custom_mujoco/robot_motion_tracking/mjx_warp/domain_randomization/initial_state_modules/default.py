import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class DefaultInitialState:
    def __init__(self, env):
        self.env = env

        self.soft_joint_position_limit = env.env_config["domain_randomization"]["initial_state"]["default"]["soft_joint_position_limit"]
        self.enable_randomization = env.env_config["domain_randomization"]["initial_state"]["default"]["enable_randomization"]
        self.joint_position = env.env_config["domain_randomization"]["initial_state"]["default"]["joint_position"]
        self.root_position = env.env_config["domain_randomization"]["initial_state"]["default"]["root_position"]
        self.root_orientation = env.env_config["domain_randomization"]["initial_state"]["default"]["root_orientation"]
        self.root_linear_velocity = env.env_config["domain_randomization"]["initial_state"]["default"]["root_linear_velocity"]
        self.root_angular_velocity = env.env_config["domain_randomization"]["initial_state"]["default"]["root_angular_velocity"]
        self.object_position = env.env_config["domain_randomization"]["initial_state"]["default"]["object_position"]
        margin = (1 - self.soft_joint_position_limit) / 2 * (env.upper - env.lower)
        self.lower, self.upper = env.lower + margin, env.upper - margin


    def sample(self, key, qpos, qvel, eval_mode):
        keys = jax.random.split(key, 6)
        enabled = self.enable_randomization & ~eval_mode
        joints = qpos[:, self.env.actuator_joint_mask_qpos] + jnp.where(enabled, jax.random.uniform(keys[0], (self.env.nr_envs, self.env.nr_actuators), minval=-self.joint_position, maxval=self.joint_position), 0.0)
        qpos = qpos.at[:, self.env.actuator_joint_mask_qpos].set(jnp.clip(joints, self.lower, self.upper))
        qpos = qpos.at[:, self.env.root_position_qpos_slice].add(jnp.where(enabled, jax.random.uniform(keys[1], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.root_position), 0.0))

        rotation = Rotation.from_quat(qpos[:, self.env.root_quaternion_qpos_slice][:, self.env.quaternion_wxyz_to_xyzw_indices])
        angular_velocity = rotation.apply(qvel[:, self.env.root_angular_qvel_slice])
        angles = jnp.where(enabled, jax.random.uniform(keys[2], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.root_orientation), 0.0)
        rotation = Rotation.from_euler("xyz", angles) * rotation
        qpos = qpos.at[:, self.env.root_quaternion_qpos_slice].set(jnp.where(enabled, rotation.as_quat()[:, self.env.quaternion_xyzw_to_wxyz_indices], qpos[:, self.env.root_quaternion_qpos_slice]))

        qvel = qvel.at[:, self.env.root_linear_qvel_slice].add(jnp.where(enabled, jax.random.uniform(keys[3], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.root_linear_velocity), 0.0))
        angular_velocity += jnp.where(enabled, jax.random.uniform(keys[4], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.root_angular_velocity), 0.0)
        qvel = qvel.at[:, self.env.root_angular_qvel_slice].set(jnp.where(enabled, rotation.inv().apply(angular_velocity), qvel[:, self.env.root_angular_qvel_slice]))

        if self.env.has_object:
            qpos = qpos.at[:, self.env.object_position_qpos_slice].add(jnp.where(enabled, jax.random.uniform(keys[5], (self.env.nr_envs, 3), minval=-1.0, maxval=1.0) * jnp.asarray(self.object_position), 0.0))
            qvel = qvel.at[:, self.env.object_angular_qvel_slice].set(0.0)

        return qpos, qvel
