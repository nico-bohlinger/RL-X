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
        keys = jax.random.split(key, 8)

        roll_angle = jax.random.uniform(keys[0], minval=-jnp.pi * self.roll_angle_pi_factor, maxval=jnp.pi * self.roll_angle_pi_factor)
        pitch_angle = jax.random.uniform(keys[1], minval=-jnp.pi * self.pitch_angle_pi_factor, maxval=jnp.pi * self.pitch_angle_pi_factor)
        yaw_angle = jax.random.uniform(keys[2], minval=-jnp.pi * self.yaw_angle_pi_factor, maxval=jnp.pi * self.yaw_angle_pi_factor)
        quaternion = Rotation.from_euler("xyz", internal_state["env_curriculum_coeff"] * jnp.array([roll_angle, pitch_angle, yaw_angle])).as_quat(scalar_first=True)
        
        actuator_joint_nominal_position_factor = internal_state["env_curriculum_coeff"] * self.actuator_joint_nominal_position_factor
        actuator_joint_positions = internal_state["actuator_joint_nominal_positions"] * jax.random.uniform(keys[3], minval=1 - actuator_joint_nominal_position_factor, maxval=1 + actuator_joint_nominal_position_factor, shape=internal_state["actuator_joint_nominal_positions"].shape)
        actuator_joint_positions += internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[4], minval=-self.actuator_joint_position_offset_to_nominal, maxval=self.actuator_joint_position_offset_to_nominal, shape=internal_state["actuator_joint_nominal_positions"].shape)
        actuator_joint_positions = jnp.clip(actuator_joint_positions, internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 0], internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 1])

        joint_velocity_max_factor = internal_state["env_curriculum_coeff"] * self.joint_velocity_max_factor
        actuator_joint_velocities = internal_state["actuator_joint_max_velocities"] * jax.random.uniform(keys[5], minval=-joint_velocity_max_factor, maxval=joint_velocity_max_factor, shape=self.env.actuator_joint_max_velocities.shape)
        
        max_trunk_velocity = jnp.minimum(jnp.sum(mjx_model.body_mass) * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)
        linear_velocities = internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[6], minval=-max_trunk_velocity, maxval=max_trunk_velocity, shape=(3,))
        angular_velocities = internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[7], minval=-max_trunk_velocity, maxval=max_trunk_velocity, shape=(3,))
        
        linear_positions = jnp.array([0.0, 0.0, internal_state["robot_nominal_qpos_height_over_ground"] + internal_state["center_height"]])

        qpos = self.env.initial_qpos
        qpos = qpos.at[:3].set(linear_positions)
        qpos = qpos.at[3:7].set(quaternion)
        qpos = qpos.at[self.env.actuator_joint_mask_qpos].set(actuator_joint_positions)

        qvel = jnp.zeros(self.env.initial_mjx_model.nv)
        qvel = qvel.at[:3].set(linear_velocities)
        qvel = qvel.at[3:6].set(angular_velocities)
        qvel = qvel.at[self.env.actuator_joint_mask_qvel].set(actuator_joint_velocities)

        data = self.env.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.env.nr_actuator_joints))
        data, _ = jax.lax.scan(
            f=lambda data_, _: (mjx.forward(mjx_model, data_), None),
            init=data,
            xs=(),
            length=1,
            unroll=True
        )
        feet_x_pos = data.geom_xpos[self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = jnp.max(self.env.terrain_function.ground_height_at(internal_state, feet_x_pos, feet_y_pos) - data.geom_xpos[self.env.foot_geom_indices, 2])
        qpos = qpos.at[2].set(qpos[2] + min_feet_z_pos_under_ground)

        return qpos, qvel
