import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx


class DefaultDRSeenRobotFunction:
    def __init__(self, env):
        self.env = env

        self.robot_size_scaling_factor = env.env_config["domain_randomization"]["seen_robot"]["robot_size_scaling_factor"]
        self.coupled_mass_inertia_factor = env.env_config["domain_randomization"]["seen_robot"]["coupled_mass_inertia_factor"]
        self.decoupled_mass_inertia_factor = env.env_config["domain_randomization"]["seen_robot"]["decoupled_mass_inertia_factor"]
        self.add_com_displacement = env.env_config["domain_randomization"]["seen_robot"]["add_com_displacement"]
        self.add_inertia_orientation_rad = env.env_config["domain_randomization"]["seen_robot"]["add_inertia_orientation_rad"]
        self.add_body_position = env.env_config["domain_randomization"]["seen_robot"]["add_body_position"]
        self.add_body_orientation_rad = env.env_config["domain_randomization"]["seen_robot"]["add_body_orientation_rad"]
        self.add_imu_position = env.env_config["domain_randomization"]["seen_robot"]["add_imu_position"]
        self.foot_size_factor = env.env_config["domain_randomization"]["seen_robot"]["foot_size_factor"]
        self.joint_axis_angle_rad = env.env_config["domain_randomization"]["seen_robot"]["joint_axis_angle_rad"]
        self.torque_limit_factor = env.env_config["domain_randomization"]["seen_robot"]["torque_limit_factor"]
        self.add_actuator_joint_nominal_position = env.env_config["domain_randomization"]["seen_robot"]["add_actuator_joint_nominal_position"]
        self.joint_velocity_max_factor = env.env_config["domain_randomization"]["seen_robot"]["joint_velocity_max_factor"]
        self.add_joint_range = env.env_config["domain_randomization"]["seen_robot"]["add_joint_range"]
        self.joint_damping_factor = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["joint_damping_factor"])
        self.add_joint_damping = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_damping"])
        self.joint_armature_factor = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["joint_armature_factor"])
        self.add_joint_armature = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_armature"])
        self.joint_stiffness_factor = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["joint_stiffness_factor"])
        self.add_joint_stiffness = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_stiffness"])
        self.joint_friction_loss_factor = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["joint_friction_loss_factor"])
        self.add_joint_friction_loss = jnp.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_friction_loss"])
        self.p_gain_factor = env.env_config["domain_randomization"]["seen_robot"]["p_gain_factor"]
        self.d_gain_factor = env.env_config["domain_randomization"]["seen_robot"]["d_gain_factor"]
        self.scaling_factor_factor = env.env_config["domain_randomization"]["seen_robot"]["scaling_factor_factor"]

        self.default_masses = self.env.initial_mjx_model.body_mass[1:]
        self.default_inertias = self.env.initial_mjx_model.body_inertia[1:]
        self.default_coms = self.env.initial_mjx_model.body_ipos[1:]
        self.default_inertia_quats_xyzw = self.env.initial_mjx_model.body_iquat[1:, [1, 2, 3, 0]]
        self.default_body_positions = self.env.initial_mjx_model.body_pos[1:]
        self.default_body_quats_xyzw = self.env.initial_mjx_model.body_quat[1:, [1, 2, 3, 0]]
        self.default_geom_pos = self.env.initial_mjx_model.geom_pos[1:]
        self.default_geom_sizes = self.env.initial_mjx_model.geom_size[1:]
        self.default_geom_rbound = self.env.initial_mjx_model.geom_rbound[1:]
        self.default_site_pos = self.env.initial_mjx_model.site_pos.copy()
        self.default_cam_pos = self.env.initial_mjx_model.cam_pos.copy()
        self.default_jnt_pos = self.env.initial_mjx_model.jnt_pos[1:]
        self.default_joint_rotation_axes = self.env.initial_mjx_model.jnt_axis[1:]
        self.default_torque_limits = self.env.initial_mjx_model.actuator_forcerange[:, 1]
        self.default_actuator_joint_nominal_positions = self.env.initial_qpos[self.env.actuator_joint_mask_qpos]
        self.default_actuator_joint_max_velocities = self.env.actuator_joint_max_velocities
        self.default_joint_ranges = self.env.initial_mjx_model.jnt_range[1:]
        self.default_joint_dampings = self.env.initial_mjx_model.dof_damping[6:]
        self.default_joint_armatures = self.env.initial_mjx_model.dof_armature[6:]
        self.default_joint_stiffnesses = self.env.initial_mjx_model.jnt_stiffness[1:]
        self.default_joint_frictionlosses = self.env.initial_mjx_model.dof_frictionloss[6:]
        self.default_p_gain = -self.env.initial_mjx_model.actuator_biasprm[0, 1]
        self.default_d_gain = -self.env.initial_mjx_model.actuator_biasprm[0, 2]
        self.default_scaling_factor = env.robot_config["scaling_factor"]

        # The perpendicular vectors for the joint axis randomization are environment-independent, so they are precomputed here
        abs_axis = jnp.abs(self.default_joint_rotation_axes)
        min_axis = jnp.argmin(abs_axis, axis=1)
        nr_axes = self.default_joint_rotation_axes.shape[0]
        perpendicular_vector = jnp.zeros_like(self.default_joint_rotation_axes)
        perpendicular_vector = perpendicular_vector.at[jnp.arange(nr_axes), (min_axis + 1) % 3].set(self.default_joint_rotation_axes[jnp.arange(nr_axes), (min_axis + 2) % 3])
        perpendicular_vector = perpendicular_vector.at[jnp.arange(nr_axes), (min_axis + 2) % 3].set(-self.default_joint_rotation_axes[jnp.arange(nr_axes), (min_axis + 1) % 3])
        self.perpendicular_vector = perpendicular_vector / jnp.linalg.norm(perpendicular_vector, axis=1, keepdims=True)
        self.second_perpendicular_vector = jnp.cross(self.default_joint_rotation_axes, self.perpendicular_vector)


    def init(self, internal_state):
        nr_envs = self.env.nr_envs
        internal_state["seen_body_masses"] = jnp.tile(self.default_masses[None], (nr_envs, 1))
        internal_state["seen_body_inertias"] = jnp.tile(self.default_inertias[None], (nr_envs, 1, 1))
        internal_state["seen_body_coms"] = jnp.tile(self.default_coms[None], (nr_envs, 1, 1))
        internal_state["seen_body_positions"] = jnp.tile(self.default_body_positions[None], (nr_envs, 1, 1))
        internal_state["seen_torque_limits"] = jnp.tile(self.default_torque_limits[None], (nr_envs, 1))
        internal_state["seen_joint_ranges"] = jnp.tile(self.default_joint_ranges[None], (nr_envs, 1, 1))
        internal_state["seen_joint_dampings"] = jnp.tile(self.default_joint_dampings[None], (nr_envs, 1))
        internal_state["seen_joint_armatures"] = jnp.tile(self.default_joint_armatures[None], (nr_envs, 1))
        internal_state["seen_joint_stiffnesses"] = jnp.tile(self.default_joint_stiffnesses[None], (nr_envs, 1))
        internal_state["seen_joint_frictionlosses"] = jnp.tile(self.default_joint_frictionlosses[None], (nr_envs, 1))
        internal_state["seen_p_gain"] = jnp.full(nr_envs, self.default_p_gain)
        internal_state["seen_d_gain"] = jnp.full(nr_envs, self.default_d_gain)
        internal_state["scaling_factor"] = jnp.full(nr_envs, self.default_scaling_factor)
        internal_state["partial_actuator_gainprm_without_dropout"] = jnp.tile(self.env.initial_mjx_model.actuator_gainprm[:, 0][None], (nr_envs, 1))
        internal_state["partial_actuator_biasprm_without_dropout"] = jnp.tile(self.env.initial_mjx_model.actuator_biasprm[:, 1:3][None], (nr_envs, 1, 1))
        internal_state["robot_nominal_qpos_height_over_ground"] = jnp.full(nr_envs, self.env.initial_qpos[2])
        internal_state["robot_nominal_imu_height_over_ground"] = jnp.full(nr_envs, self.env.initial_imu_height)
        internal_state["nr_collisions_in_nominal"] = jnp.zeros(nr_envs)


    def sample(self, internal_state, mjx_model, data, should_randomize, key):
        nr_envs = self.env.nr_envs
        keys = jax.random.split(key, 29)

        # During evaluation, we don't want to randomize the (seen) robot parameters
        env_curriculum_coeff = internal_state["env_curriculum_coeff"] * jnp.where(internal_state["in_eval_mode"], 0.0, 1.0)
        cc = env_curriculum_coeff[:, None]

        body_size_factor = 1 + cc * jax.random.uniform(keys[0], minval=-self.robot_size_scaling_factor, maxval=self.robot_size_scaling_factor, shape=(nr_envs, 3))
        body_size_factor = jnp.where(self.env.has_equality_constraints, jnp.array([1.0, 1.0, 1.0]), body_size_factor)
        avg_body_size_factor = jnp.mean(body_size_factor, axis=-1)
        default_masses = self.default_masses[None] * jnp.power(avg_body_size_factor, 3)[:, None]
        default_inertias = self.default_inertias[None] * jnp.power(body_size_factor, 5)[:, None, :]
        default_coms = self.default_coms[None] * body_size_factor[:, None, :]
        default_body_positions = self.default_body_positions[None] * body_size_factor[:, None, :]
        default_site_positions = self.default_site_pos[None] * body_size_factor[:, None, :]
        default_geom_sizes = self.default_geom_sizes[None] * avg_body_size_factor[:, None, None]
        geom_positions = jnp.broadcast_to(self.env.initial_mjx_model.geom_pos[None], (nr_envs,) + self.env.initial_mjx_model.geom_pos.shape)
        geom_positions = geom_positions.at[:, 1:].set(self.default_geom_pos[None] * body_size_factor[:, None, :])
        geom_rbounds = jnp.broadcast_to(self.env.initial_mjx_model.geom_rbound[None], (nr_envs,) + self.env.initial_mjx_model.geom_rbound.shape)
        geom_rbounds = geom_rbounds.at[:, 1:].set(self.default_geom_rbound[None] * avg_body_size_factor[:, None])
        camera_positions = self.default_cam_pos[None] * body_size_factor[:, None, :]
        joint_positions = jnp.broadcast_to(self.env.initial_mjx_model.jnt_pos[None], (nr_envs,) + self.env.initial_mjx_model.jnt_pos.shape)
        joint_positions = joint_positions.at[:, 1:].set(self.default_jnt_pos[None] * body_size_factor[:, None, :])
        default_torque_limits = self.default_torque_limits[None] * avg_body_size_factor[:, None]
        default_actuator_joint_max_velocities = self.default_actuator_joint_max_velocities[None] * avg_body_size_factor[:, None]
        default_joint_dampings = self.default_joint_dampings[None] * avg_body_size_factor[:, None]
        default_joint_armatures = self.default_joint_armatures[None] * avg_body_size_factor[:, None]
        default_joint_stiffnesses = self.default_joint_stiffnesses[None] * avg_body_size_factor[:, None]
        default_joint_frictionlosses = self.default_joint_frictionlosses[None] * avg_body_size_factor[:, None]
        default_p_gain = self.default_p_gain * avg_body_size_factor
        default_d_gain = self.default_d_gain * avg_body_size_factor
        default_scaling_factor = self.default_scaling_factor * avg_body_size_factor

        coupled_masses = default_masses * (1 + cc * jax.random.uniform(keys[1], minval=-self.coupled_mass_inertia_factor, maxval=self.coupled_mass_inertia_factor, shape=default_masses.shape))
        coupled_inertias = default_inertias * jnp.reshape(coupled_masses / default_masses, coupled_masses.shape + (1,))
        seen_body_masses = coupled_masses * (1 + cc * jax.random.uniform(keys[2], minval=-self.decoupled_mass_inertia_factor, maxval=self.decoupled_mass_inertia_factor, shape=coupled_masses.shape))
        seen_inertias = coupled_inertias * (1 + cc[:, :, None] * jax.random.uniform(keys[3], minval=-self.decoupled_mass_inertia_factor, maxval=self.decoupled_mass_inertia_factor, shape=coupled_inertias.shape))
        masses = seen_body_masses * internal_state["mass_inertia_noise_factors"]
        inertias = seen_inertias * internal_state["mass_inertia_noise_factors"][:, :, None]
        body_mass = jnp.broadcast_to(self.env.initial_mjx_model.body_mass[None], (nr_envs,) + self.env.initial_mjx_model.body_mass.shape).at[:, 1:].set(masses)
        body_inertia = jnp.broadcast_to(self.env.initial_mjx_model.body_inertia[None], (nr_envs,) + self.env.initial_mjx_model.body_inertia.shape).at[:, 1:].set(inertias)

        seen_coms = default_coms + cc[:, :, None] * jax.random.uniform(keys[4], minval=-self.add_com_displacement, maxval=self.add_com_displacement, shape=default_coms.shape)
        coms = seen_coms * internal_state["com_noise_factors"]
        body_ipos = jnp.broadcast_to(self.env.initial_mjx_model.body_ipos[None], (nr_envs,) + self.env.initial_mjx_model.body_ipos.shape).at[:, 1:].set(coms)

        axes = jax.random.normal(keys[5], shape=(nr_envs,) + self.default_inertia_quats_xyzw.shape[:1] + (3,))
        axes /= jnp.linalg.norm(axes, axis=2, keepdims=True)
        noisy_angles = cc * jax.random.uniform(keys[6], minval=-self.add_inertia_orientation_rad, maxval=self.add_inertia_orientation_rad, shape=(nr_envs,) + self.default_inertia_quats_xyzw.shape[:1])
        noisy_rotations = Rotation.from_rotvec(axes * noisy_angles[:, :, None])
        default_rotations = Rotation.from_quat(jnp.broadcast_to(self.default_inertia_quats_xyzw[None], (nr_envs,) + self.default_inertia_quats_xyzw.shape))
        inertia_quats = (noisy_rotations * default_rotations).as_quat(scalar_first=True)
        body_iquat = jnp.broadcast_to(self.env.initial_mjx_model.body_iquat[None], (nr_envs,) + self.env.initial_mjx_model.body_iquat.shape).at[:, 1:].set(inertia_quats)

        seen_body_positions = default_body_positions + cc[:, :, None] * jax.random.uniform(keys[7], minval=-self.add_body_position, maxval=self.add_body_position, shape=default_body_positions.shape)
        seen_body_positions = jnp.where(self.env.has_equality_constraints, default_body_positions, seen_body_positions)
        body_positions = seen_body_positions * internal_state["body_position_noise_factors"]
        body_positions = jnp.broadcast_to(self.env.initial_mjx_model.body_pos[None], (nr_envs,) + self.env.initial_mjx_model.body_pos.shape).at[:, 1:].set(body_positions)

        axes = jax.random.normal(keys[8], shape=(nr_envs,) + self.default_body_quats_xyzw.shape[:1] + (3,))
        axes /= jnp.linalg.norm(axes, axis=2, keepdims=True)
        noisy_angles = cc * jax.random.uniform(keys[9], minval=-self.add_body_orientation_rad, maxval=self.add_body_orientation_rad, shape=(nr_envs,) + self.default_body_quats_xyzw.shape[:1])
        noisy_angles = jnp.where(self.env.has_equality_constraints, jnp.zeros_like(noisy_angles), noisy_angles)
        noisy_rotations = Rotation.from_rotvec(axes * noisy_angles[:, :, None])
        default_rotations = Rotation.from_quat(jnp.broadcast_to(self.default_body_quats_xyzw[None], (nr_envs,) + self.default_body_quats_xyzw.shape))
        body_quats = (noisy_rotations * default_rotations).as_quat(scalar_first=True)
        body_quats = jnp.broadcast_to(self.env.initial_mjx_model.body_quat[None], (nr_envs,) + self.env.initial_mjx_model.body_quat.shape).at[:, 1:].set(body_quats)

        site_positions = default_site_positions.at[:, self.env.imu_site_id].set(default_site_positions[:, self.env.imu_site_id] + cc * jax.random.uniform(keys[10], minval=-self.add_imu_position, maxval=self.add_imu_position, shape=(nr_envs, 3)))

        geom_sizes = default_geom_sizes.at[:, self.env.foot_geom_indices - 1].set(default_geom_sizes[:, self.env.foot_geom_indices - 1] * (1 + cc[:, :, None] * jax.random.uniform(keys[11], minval=-self.foot_size_factor, maxval=self.foot_size_factor, shape=(nr_envs, self.env.foot_geom_indices.shape[0], 3))))
        geom_sizes = jnp.broadcast_to(self.env.initial_mjx_model.geom_size[None], (nr_envs,) + self.env.initial_mjx_model.geom_size.shape).at[:, 1:].set(geom_sizes)

        nr_axes = self.default_joint_rotation_axes.shape[0]
        phi = jax.random.uniform(keys[12], minval=0, maxval=2 * jnp.pi, shape=(nr_envs, nr_axes, 1))
        random_axis = jnp.cos(phi) * self.perpendicular_vector[None] + jnp.sin(phi) * self.second_perpendicular_vector[None]
        random_angle = cc[:, :, None] * jax.random.uniform(keys[13], minval=0, maxval=self.joint_axis_angle_rad, shape=(nr_envs, nr_axes, 1))
        rotation = Rotation.from_rotvec(random_angle * random_axis)
        rotated_joint_rotation_axes = rotation.apply(jnp.broadcast_to(self.default_joint_rotation_axes[None], (nr_envs, nr_axes, 3)))
        jnt_axis = jnp.broadcast_to(self.env.initial_mjx_model.jnt_axis[None], (nr_envs,) + self.env.initial_mjx_model.jnt_axis.shape).at[:, 1:].set(rotated_joint_rotation_axes)

        torque_limits = default_torque_limits * (1 + cc * jax.random.uniform(keys[14], minval=-self.torque_limit_factor, maxval=self.torque_limit_factor, shape=default_torque_limits.shape))
        actuators_forcerange = jnp.broadcast_to(self.env.initial_mjx_model.actuator_forcerange[None], (nr_envs,) + self.env.initial_mjx_model.actuator_forcerange.shape)
        actuators_forcerange = actuators_forcerange.at[:, :, 1].set(torque_limits)
        actuators_forcerange = actuators_forcerange.at[:, :, 0].set(-torque_limits)

        actuator_joint_nominal_positions = self.default_actuator_joint_nominal_positions[None] + cc * jax.random.uniform(keys[15], minval=-self.add_actuator_joint_nominal_position, maxval=self.add_actuator_joint_nominal_position, shape=(nr_envs,) + self.default_actuator_joint_nominal_positions.shape)
        actuator_joint_nominal_positions = jnp.clip(actuator_joint_nominal_positions, internal_state["joint_position_limits"][:, self.env.actuator_joint_mask_joints - 1, 0], internal_state["joint_position_limits"][:, self.env.actuator_joint_mask_joints - 1, 1])

        actuator_joint_max_velocities = default_actuator_joint_max_velocities * (1 + cc * jax.random.uniform(keys[16], minval=-self.joint_velocity_max_factor, maxval=self.joint_velocity_max_factor, shape=default_actuator_joint_max_velocities.shape))

        joint_ranges = self.default_joint_ranges[None] + cc[:, :, None] * jax.random.uniform(keys[17], minval=-self.add_joint_range, maxval=self.add_joint_range, shape=(nr_envs,) + self.default_joint_ranges.shape)
        jnt_range = jnp.broadcast_to(self.env.initial_mjx_model.jnt_range[None], (nr_envs,) + self.env.initial_mjx_model.jnt_range.shape).at[:, 1:].set(joint_ranges)

        seen_joint_dampings = default_joint_dampings * (1 + cc * jax.random.uniform(keys[18], minval=-self.joint_damping_factor, maxval=self.joint_damping_factor, shape=default_joint_dampings.shape))
        seen_joint_dampings += cc * jax.random.uniform(keys[19], minval=-self.add_joint_damping, maxval=self.add_joint_damping, shape=default_joint_dampings.shape)
        seen_joint_dampings = jnp.maximum(0, seen_joint_dampings)
        dof_damping = jnp.broadcast_to(self.env.initial_mjx_model.dof_damping[None], (nr_envs,) + self.env.initial_mjx_model.dof_damping.shape).at[:, 6:].set(seen_joint_dampings * internal_state["joint_damping_noise_factors"])

        seen_joint_armatures = default_joint_armatures * (1 + cc * jax.random.uniform(keys[20], minval=-self.joint_armature_factor, maxval=self.joint_armature_factor, shape=default_joint_armatures.shape))
        seen_joint_armatures += cc * jax.random.uniform(keys[21], minval=-self.add_joint_armature, maxval=self.add_joint_armature, shape=default_joint_armatures.shape)
        seen_joint_armatures = jnp.maximum(0, seen_joint_armatures)
        dof_armature = jnp.broadcast_to(self.env.initial_mjx_model.dof_armature[None], (nr_envs,) + self.env.initial_mjx_model.dof_armature.shape).at[:, 6:].set(seen_joint_armatures * internal_state["joint_armature_noise_factors"])

        seen_joint_stiffnesses = default_joint_stiffnesses * (1 + cc * jax.random.uniform(keys[22], minval=-self.joint_stiffness_factor, maxval=self.joint_stiffness_factor, shape=default_joint_stiffnesses.shape))
        seen_joint_stiffnesses += cc * jax.random.uniform(keys[23], minval=-self.add_joint_stiffness, maxval=self.add_joint_stiffness, shape=default_joint_stiffnesses.shape)
        seen_joint_stiffnesses = jnp.maximum(0, seen_joint_stiffnesses)
        jnt_stiffness = jnp.broadcast_to(self.env.initial_mjx_model.jnt_stiffness[None], (nr_envs,) + self.env.initial_mjx_model.jnt_stiffness.shape).at[:, 1:].set(seen_joint_stiffnesses * internal_state["joint_stiffness_noise_factors"])

        seen_joint_frictionlosses = default_joint_frictionlosses * (1 + cc * jax.random.uniform(keys[24], minval=-self.joint_friction_loss_factor, maxval=self.joint_friction_loss_factor, shape=default_joint_frictionlosses.shape))
        seen_joint_frictionlosses += cc * jax.random.uniform(keys[25], minval=-self.add_joint_friction_loss, maxval=self.add_joint_friction_loss, shape=default_joint_frictionlosses.shape)
        seen_joint_frictionlosses = jnp.maximum(0, seen_joint_frictionlosses)
        dof_frictionloss = jnp.broadcast_to(self.env.initial_mjx_model.dof_frictionloss[None], (nr_envs,) + self.env.initial_mjx_model.dof_frictionloss.shape).at[:, 6:].set(seen_joint_frictionlosses * internal_state["joint_friction_loss_noise_factors"])

        seen_p_gain = default_p_gain * (1 + env_curriculum_coeff * jax.random.uniform(keys[26], (nr_envs,), minval=-self.p_gain_factor, maxval=self.p_gain_factor))
        p_gain = seen_p_gain[:, None] * internal_state["p_gain_noise_factors"]
        seen_d_gain = default_d_gain * (1 + env_curriculum_coeff * jax.random.uniform(keys[27], (nr_envs,), minval=-self.d_gain_factor, maxval=self.d_gain_factor))
        d_gain = seen_d_gain[:, None] * internal_state["d_gain_noise_factors"]
        scaling_factor = default_scaling_factor * (1 + env_curriculum_coeff * jax.random.uniform(keys[28], (nr_envs,), minval=-self.scaling_factor_factor, maxval=self.scaling_factor_factor))
        actuators_gainprm = jnp.broadcast_to(self.env.initial_mjx_model.actuator_gainprm[None], (nr_envs,) + self.env.initial_mjx_model.actuator_gainprm.shape).at[:, :, 0].set(p_gain)
        actuators_biasprm = jnp.broadcast_to(self.env.initial_mjx_model.actuator_biasprm[None], (nr_envs,) + self.env.initial_mjx_model.actuator_biasprm.shape)
        actuators_biasprm = actuators_biasprm.at[:, :, 1].set(-p_gain)
        actuators_biasprm = actuators_biasprm.at[:, :, 2].set(-d_gain)

        new_fields = {
            "body_pos": body_positions,
            "body_quat": body_quats,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "body_ipos": body_ipos,
            "body_iquat": body_iquat,
            "site_pos": site_positions,
            "geom_size": geom_sizes,
            "geom_pos": geom_positions,
            "geom_rbound": geom_rbounds,
            "cam_pos": camera_positions,
            "jnt_pos": joint_positions,
            "jnt_axis": jnt_axis,
            "actuator_forcerange": actuators_forcerange,
            "jnt_range": jnt_range,
            "dof_damping": dof_damping,
            "dof_armature": dof_armature,
            "jnt_stiffness": jnt_stiffness,
            "dof_frictionloss": dof_frictionloss,
            "actuator_gainprm": actuators_gainprm,
            "actuator_biasprm": actuators_biasprm,
        }
        new_mjx_model = mjx_model.tree_replace(new_fields)
        merged_fields = {}
        for field_name, new_value in new_fields.items():
            old_value = getattr(mjx_model, field_name)
            merged_fields[field_name] = jnp.where(should_randomize.reshape((nr_envs,) + (1,) * (new_value.ndim - 1)), new_value, old_value)
        mjx_model = mjx_model.tree_replace(merged_fields)

        internal_state["seen_body_masses"] = jnp.where(should_randomize[:, None], seen_body_masses, internal_state["seen_body_masses"])
        internal_state["seen_body_inertias"] = jnp.where(should_randomize[:, None, None], seen_inertias, internal_state["seen_body_inertias"])
        internal_state["seen_body_coms"] = jnp.where(should_randomize[:, None, None], seen_coms, internal_state["seen_body_coms"])
        internal_state["seen_body_positions"] = jnp.where(should_randomize[:, None, None], seen_body_positions, internal_state["seen_body_positions"])
        internal_state["actuator_joint_nominal_positions"] = jnp.where(should_randomize[:, None], actuator_joint_nominal_positions, internal_state["actuator_joint_nominal_positions"])
        internal_state["actuator_joint_max_velocities"] = jnp.where(should_randomize[:, None], actuator_joint_max_velocities, internal_state["actuator_joint_max_velocities"])
        internal_state["seen_joint_ranges"] = jnp.where(should_randomize[:, None, None], joint_ranges, internal_state["seen_joint_ranges"])
        internal_state["seen_joint_dampings"] = jnp.where(should_randomize[:, None], seen_joint_dampings, internal_state["seen_joint_dampings"])
        internal_state["seen_joint_armatures"] = jnp.where(should_randomize[:, None], seen_joint_armatures, internal_state["seen_joint_armatures"])
        internal_state["seen_joint_stiffnesses"] = jnp.where(should_randomize[:, None], seen_joint_stiffnesses, internal_state["seen_joint_stiffnesses"])
        internal_state["seen_joint_frictionlosses"] = jnp.where(should_randomize[:, None], seen_joint_frictionlosses, internal_state["seen_joint_frictionlosses"])
        internal_state["seen_p_gain"] = jnp.where(should_randomize, seen_p_gain, internal_state["seen_p_gain"])
        internal_state["seen_d_gain"] = jnp.where(should_randomize, seen_d_gain, internal_state["seen_d_gain"])
        internal_state["scaling_factor"] = jnp.where(should_randomize, scaling_factor, internal_state["scaling_factor"])
        internal_state["partial_actuator_gainprm_without_dropout"] = jnp.where(should_randomize[:, None], mjx_model.actuator_gainprm[:, :, 0], internal_state["partial_actuator_gainprm_without_dropout"])
        internal_state["partial_actuator_biasprm_without_dropout"] = jnp.where(should_randomize[:, None, None], mjx_model.actuator_biasprm[:, :, 1:3], internal_state["partial_actuator_biasprm_without_dropout"])

        qpos = jnp.tile(self.env.initial_qpos[None], (nr_envs, 1))
        qpos = qpos.at[:, self.env.actuator_joint_mask_qpos].set(actuator_joint_nominal_positions)
        qpos = qpos.at[:, 2].set(qpos[:, 2] + internal_state["center_height"])
        qvel = jnp.zeros((nr_envs, self.env.initial_mj_model.nv))
        data_tmp = self.env.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((nr_envs, self.env.nr_actuator_joints)))
        data_tmp = mjx.forward(new_mjx_model, data_tmp)
        min_feet_z_pos = jnp.min(data_tmp.geom_xpos[:, self.env.foot_geom_indices, 2], axis=-1)
        offset = internal_state["center_height"] - min_feet_z_pos
        robot_nominal_qpos_height_over_ground = qpos[:, 2] - internal_state["center_height"] + offset
        robot_nominal_imu_height_over_ground = data_tmp.site_xpos[:, self.env.imu_site_id, 2] - internal_state["center_height"] + offset
        internal_state["robot_nominal_qpos_height_over_ground"] = jnp.where(should_randomize, robot_nominal_qpos_height_over_ground, internal_state["robot_nominal_qpos_height_over_ground"])
        internal_state["robot_nominal_imu_height_over_ground"] = jnp.where(should_randomize, robot_nominal_imu_height_over_ground, internal_state["robot_nominal_imu_height_over_ground"])
        all_contact_relevant_geom_xpos = data_tmp.geom_xpos[:, self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = new_mjx_model.geom_size[:, self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = jnp.linalg.norm(all_contact_relevant_geom_xpos[:, :, None] - all_contact_relevant_geom_xpos[:, None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, :, None] + all_contact_relevant_geom_sizes[:, None])
        nr_collisions = (jnp.sum(contact_between_geoms, axis=(1, 2)) - self.env.reward_collision_sphere_geom_ids.shape[0]) // 2
        internal_state["nr_collisions_in_nominal"] = jnp.where(should_randomize, nr_collisions, internal_state["nr_collisions_in_nominal"])

        data_tmp = data_tmp.replace(qpos=data.qpos)
        data_tmp = mjx.forward(new_mjx_model, data_tmp)
        feet_x_pos = data_tmp.geom_xpos[:, self.env.foot_geom_indices, 0]
        feet_y_pos = data_tmp.geom_xpos[:, self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = jnp.max(self.env.terrain_function.ground_height_at(internal_state, feet_x_pos, feet_y_pos) - data_tmp.geom_xpos[:, self.env.foot_geom_indices, 2], axis=-1)
        data = data.replace(qpos=jnp.where(should_randomize[:, None], data.qpos.at[:, 2].set(data.qpos[:, 2] + min_feet_z_pos_under_ground), data.qpos))

        return mjx_model, data
