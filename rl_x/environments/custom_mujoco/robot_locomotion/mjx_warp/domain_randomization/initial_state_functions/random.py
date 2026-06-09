import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx


class RandomDRInitialState:
    def __init__(self, env):
        self.env = env

        self.roll_angle_pi_factor = env.env_config["domain_randomization"]["initial_state"]["roll_angle_pi_factor"]
        self.pitch_angle_pi_factor = env.env_config["domain_randomization"]["initial_state"]["pitch_angle_pi_factor"]
        self.yaw_angle_pi_factor = env.env_config["domain_randomization"]["initial_state"]["yaw_angle_pi_factor"]
        self.actuator_joint_position_offset_to_nominal = env.env_config["domain_randomization"]["initial_state"]["actuator_joint_position_offset_to_nominal"]
        self.actuator_joint_nominal_position_factor = env.env_config["domain_randomization"]["initial_state"]["actuator_joint_nominal_position_factor"]
        self.joint_velocity_max_factor = env.env_config["domain_randomization"]["initial_state"]["joint_velocity_max_factor"]
        self.trunk_velocity_clip_mass_factor = env.env_config["domain_randomization"]["initial_state"]["trunk_velocity_clip_mass_factor"]
        self.trunk_velocity_clip_limit = env.env_config["domain_randomization"]["initial_state"]["trunk_velocity_clip_limit"]


    def setup(self, mjx_model, internal_state, key):
        nr_envs = self.env.nr_envs
        keys = jax.random.split(key, 8)
        curriculum_coeff = internal_state["env_curriculum_coeff"][:, None]

        roll_angle = jax.random.uniform(keys[0], (nr_envs,), minval=-jnp.pi * self.roll_angle_pi_factor, maxval=jnp.pi * self.roll_angle_pi_factor)
        pitch_angle = jax.random.uniform(keys[1], (nr_envs,), minval=-jnp.pi * self.pitch_angle_pi_factor, maxval=jnp.pi * self.pitch_angle_pi_factor)
        yaw_angle = jax.random.uniform(keys[2], (nr_envs,), minval=-jnp.pi * self.yaw_angle_pi_factor, maxval=jnp.pi * self.yaw_angle_pi_factor)
        quaternion = Rotation.from_euler("xyz", curriculum_coeff * jnp.stack([roll_angle, pitch_angle, yaw_angle], axis=-1)).as_quat(scalar_first=True)

        actuator_joint_nominal_position_factor = curriculum_coeff * self.actuator_joint_nominal_position_factor
        actuator_joint_positions = internal_state["actuator_joint_nominal_positions"] * jax.random.uniform(keys[3], minval=1 - actuator_joint_nominal_position_factor, maxval=1 + actuator_joint_nominal_position_factor, shape=internal_state["actuator_joint_nominal_positions"].shape)
        actuator_joint_positions += curriculum_coeff * jax.random.uniform(keys[4], minval=-self.actuator_joint_position_offset_to_nominal, maxval=self.actuator_joint_position_offset_to_nominal, shape=internal_state["actuator_joint_nominal_positions"].shape)
        actuator_joint_positions = jnp.clip(actuator_joint_positions, internal_state["joint_position_limits"][:, self.env.actuator_joint_mask_joints - 1, 0], internal_state["joint_position_limits"][:, self.env.actuator_joint_mask_joints - 1, 1])

        joint_velocity_max_factor = curriculum_coeff * self.joint_velocity_max_factor
        actuator_joint_velocities = internal_state["actuator_joint_max_velocities"] * jax.random.uniform(keys[5], minval=-joint_velocity_max_factor, maxval=joint_velocity_max_factor, shape=internal_state["actuator_joint_max_velocities"].shape)

        total_mass = jnp.sum(mjx_model.body_mass, axis=-1)
        max_trunk_velocity = jnp.minimum(total_mass * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)
        if jnp.ndim(max_trunk_velocity) == 0:
            max_trunk_velocity = jnp.full(nr_envs, max_trunk_velocity)
        max_trunk_velocity = max_trunk_velocity[:, None]
        linear_velocities = curriculum_coeff * jax.random.uniform(keys[6], (nr_envs, 3), minval=-max_trunk_velocity, maxval=max_trunk_velocity)
        angular_velocities = curriculum_coeff * jax.random.uniform(keys[7], (nr_envs, 3), minval=-max_trunk_velocity, maxval=max_trunk_velocity)

        linear_positions = jnp.stack([jnp.zeros(nr_envs), jnp.zeros(nr_envs), internal_state["robot_nominal_qpos_height_over_ground"] + internal_state["center_height"]], axis=-1)

        qpos = jnp.tile(self.env.initial_qpos[None], (nr_envs, 1))
        qpos = qpos.at[:, :3].set(linear_positions)
        qpos = qpos.at[:, 3:7].set(quaternion)
        qpos = qpos.at[:, self.env.actuator_joint_mask_qpos].set(actuator_joint_positions)

        qvel = jnp.zeros((nr_envs, self.env.initial_mj_model.nv))
        qvel = qvel.at[:, :3].set(linear_velocities)
        qvel = qvel.at[:, 3:6].set(angular_velocities)
        qvel = qvel.at[:, self.env.actuator_joint_mask_qvel].set(actuator_joint_velocities)

        data = self.env.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((nr_envs, self.env.nr_actuator_joints)))
        data = mjx.forward(mjx_model, data)
        feet_x_pos = data.geom_xpos[:, self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[:, self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = jnp.max(self.env.terrain_function.ground_height_at(internal_state, feet_x_pos, feet_y_pos) - data.geom_xpos[:, self.env.foot_geom_indices, 2], axis=-1)
        qpos = qpos.at[:, 2].set(qpos[:, 2] + min_feet_z_pos_under_ground)

        return qpos, qvel
