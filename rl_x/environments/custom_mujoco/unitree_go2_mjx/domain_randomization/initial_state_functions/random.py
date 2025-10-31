import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx


class RandomDRInitialState:
    def __init__(self, env):
        self.env = env

        self.roll_angle_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["roll_angle_factor"]
        self.pitch_angle_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["pitch_angle_factor"]
        self.yaw_angle_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["yaw_angle_factor"]
        self.nominal_joint_position_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["nominal_joint_position_factor"]
        self.joint_velocity_factor = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["joint_velocity_factor"]
        self.max_linear_velocity = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["max_linear_velocity"]
        self.max_angular_velocity = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["initial_state"]["max_angular_velocity"]

        self.initial_qpos = jnp.array(self.env.initial_mj_model.keyframe("home").qpos)


    def setup(self, data, mjx_model, internal_state, key):
        keys = jax.random.split(key, 7)

        roll_angle = jax.random.uniform(keys[0], minval=-jnp.pi * self.roll_angle_factor, maxval=jnp.pi * self.roll_angle_factor)
        pitch_angle = jax.random.uniform(keys[1], minval=-jnp.pi * self.pitch_angle_factor, maxval=jnp.pi * self.pitch_angle_factor)
        yaw_angle = jax.random.uniform(keys[2], minval=-jnp.pi * self.yaw_angle_factor, maxval=jnp.pi * self.yaw_angle_factor)
        quaternion = Rotation.from_euler("xyz", jnp.array([roll_angle, pitch_angle, yaw_angle])).as_quat()
        quaternion = jnp.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        joint_positions = self.env.joint_nominal_positions * jax.random.uniform(keys[3], minval=self.nominal_joint_position_factor, maxval=1 + self.nominal_joint_position_factor, shape=self.env.joint_nominal_positions.shape)
        joint_velocities = self.env.joint_max_velocities * jax.random.uniform(keys[4], minval=-self.joint_velocity_factor, maxval=self.joint_velocity_factor, shape=self.env.joint_max_velocities.shape)
        linear_velocities = jax.random.uniform(keys[5], minval=-self.max_linear_velocity, maxval=self.max_linear_velocity, shape=(3,))
        angular_velocities = jax.random.uniform(keys[6], minval=-self.max_angular_velocity, maxval=self.max_angular_velocity, shape=(3,))
        
        linear_positions = jnp.array([0.0, 0.0, self.initial_qpos[2]])

        qpos = jnp.concatenate([linear_positions, quaternion, joint_positions])
        qvel = jnp.concatenate([linear_velocities, angular_velocities, joint_velocities])

        tmp_data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.env.initial_mjx_model.nu))
        data, _ = jax.lax.scan(
            f=lambda data_, _: (mjx.step(mjx_model, data_), None),
            init=tmp_data,
            xs=(),
            length=1
        )
        min_feet_z_pos = jnp.min(data.geom_xpos[self.env.foot_geom_indices, 2])
        offset = jnp.maximum(internal_state["center_height"], -min_feet_z_pos)
        qpos = qpos.at[2].set(qpos[2] + offset + internal_state["center_height"])

        return qpos, qvel
