import numpy as np
from scipy.spatial.transform import Rotation
import mujoco


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
        self.joint_damping_factor = np.array(env.env_config["domain_randomization"]["seen_robot"]["joint_damping_factor"])
        self.add_joint_damping = np.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_damping"])
        self.joint_armature_factor = np.array(env.env_config["domain_randomization"]["seen_robot"]["joint_armature_factor"])
        self.add_joint_armature = np.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_armature"])
        self.joint_stiffness_factor = np.array(env.env_config["domain_randomization"]["seen_robot"]["joint_stiffness_factor"])
        self.add_joint_stiffness = np.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_stiffness"])
        self.joint_friction_loss_factor = np.array(env.env_config["domain_randomization"]["seen_robot"]["joint_friction_loss_factor"])
        self.add_joint_friction_loss = np.array(env.env_config["domain_randomization"]["seen_robot"]["add_joint_friction_loss"])
        self.p_gain_factor = env.env_config["domain_randomization"]["seen_robot"]["p_gain_factor"]
        self.d_gain_factor = env.env_config["domain_randomization"]["seen_robot"]["d_gain_factor"]
        self.scaling_factor_factor = env.env_config["domain_randomization"]["seen_robot"]["scaling_factor_factor"]

        self.default_masses = self.env.initial_mj_model.body_mass[1:]
        self.default_inertias = self.env.initial_mj_model.body_inertia[1:]
        self.default_coms = self.env.initial_mj_model.body_ipos[1:]
        self.default_inertia_quats = self.env.initial_mj_model.body_iquat[1:]
        self.default_body_positions = self.env.initial_mj_model.body_pos[1:]
        self.default_body_quats = self.env.initial_mj_model.body_quat[1:]
        self.default_geom_pos = self.env.initial_mj_model.geom_pos[1:]
        self.default_geom_sizes = self.env.initial_mj_model.geom_size[1:]
        self.default_geom_rbound = self.env.initial_mj_model.geom_rbound[1:]
        self.default_geom_aabb = self.env.initial_mj_model.geom_aabb[1:]
        self.default_mesh_pos = self.env.initial_mj_model.mesh_pos.copy()
        self.default_mesh_vert = self.env.initial_mj_model.mesh_vert.copy()
        self.default_site_pos = self.env.initial_mj_model.site_pos.copy()
        self.default_site_size = self.env.initial_mj_model.site_size.copy()
        self.default_cam_pos = self.env.initial_mj_model.cam_pos.copy()
        self.default_jnt_pos = self.env.initial_mj_model.jnt_pos[1:]
        self.default_joint_rotation_axes = self.env.initial_mj_model.jnt_axis[1:]
        self.default_torque_limits = self.env.initial_mj_model.actuator_forcerange[:, 1]
        self.default_actuator_joint_nominal_positions = self.env.initial_qpos[self.env.actuator_joint_mask_qpos]
        self.default_actuator_joint_max_velocities = self.env.actuator_joint_max_velocities
        self.default_joint_ranges = self.env.initial_mj_model.jnt_range[1:]
        self.default_joint_dampings = self.env.initial_mj_model.dof_damping[6:]
        self.default_joint_armatures = self.env.initial_mj_model.dof_armature[6:]
        self.default_joint_stiffnesses = self.env.initial_mj_model.jnt_stiffness[1:]
        self.default_joint_frictionlosses = self.env.initial_mj_model.dof_frictionloss[6:]
        self.default_p_gain = -self.env.initial_mj_model.actuator_biasprm[0, 1]
        self.default_d_gain = -self.env.initial_mj_model.actuator_biasprm[0, 2]
        self.default_scaling_factor = env.robot_config["scaling_factor"]


    def init(self):
        self.env.internal_state["seen_body_masses"] = self.default_masses
        self.env.internal_state["seen_body_inertias"] = self.default_inertias
        self.env.internal_state["seen_body_coms"] = self.default_coms
        self.env.internal_state["seen_body_positions"] = self.default_body_positions
        self.env.internal_state["seen_torque_limits"] = self.default_torque_limits
        self.env.internal_state["seen_joint_ranges"] = self.default_joint_ranges
        self.env.internal_state["seen_joint_dampings"] = self.default_joint_dampings
        self.env.internal_state["seen_joint_armatures"] = self.default_joint_armatures
        self.env.internal_state["seen_joint_stiffnesses"] = self.default_joint_stiffnesses
        self.env.internal_state["seen_joint_frictionlosses"] = self.default_joint_frictionlosses
        self.env.internal_state["seen_p_gain"] = self.default_p_gain
        self.env.internal_state["seen_d_gain"] = self.default_d_gain
        self.env.internal_state["scaling_factor"] = self.default_scaling_factor
        self.env.internal_state["partial_actuator_gainprm_without_dropout"] = self.env.initial_mj_model.actuator_gainprm[:, 0]
        self.env.internal_state["partial_actuator_biasprm_without_dropout"] = self.env.initial_mj_model.actuator_biasprm[:, 1:3]
        self.env.internal_state["robot_nominal_qpos_height_over_ground"] = self.env.initial_qpos[2]
        self.env.internal_state["robot_nominal_imu_height_over_ground"] = self.env.initial_imu_height


    def sample(self):
        # During evaluation, we don't want to randomize the (seen) robot parameters
        env_curriculum_coeff = self.env.internal_state["env_curriculum_coeff"] * np.where(self.env.internal_state["in_eval_mode"], 0.0, 1.0)

        body_size_factor = 1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.robot_size_scaling_factor, high=self.robot_size_scaling_factor, size=(3,))
        body_size_factor = np.where(self.env.has_equality_constraints, np.array([1.0, 1.0, 1.0]), body_size_factor)
        avg_body_size_factor = np.mean(body_size_factor)
        default_masses = self.default_masses * np.power(avg_body_size_factor, 3)
        default_inertias = self.default_inertias * np.power(body_size_factor, 5)
        default_coms = self.default_coms * body_size_factor
        default_body_positions = self.default_body_positions * body_size_factor
        default_site_positions = self.default_site_pos * body_size_factor
        default_geom_sizes = self.default_geom_sizes * avg_body_size_factor
        geom_positions = self.default_geom_pos * body_size_factor
        geom_rbounds = self.default_geom_rbound * avg_body_size_factor
        geom_aabbs = self.default_geom_aabb * np.tile(body_size_factor, 2)
        mesh_positions = self.default_mesh_pos * body_size_factor
        mesh_verts = self.default_mesh_vert * body_size_factor
        site_sizes = self.default_site_size * avg_body_size_factor
        camera_positions = self.default_cam_pos * body_size_factor
        joint_positions = self.default_jnt_pos * body_size_factor
        default_torque_limits = self.default_torque_limits * avg_body_size_factor
        default_actuator_joint_max_velocities = self.default_actuator_joint_max_velocities * avg_body_size_factor
        default_joint_dampings = self.default_joint_dampings * avg_body_size_factor
        default_joint_armatures = self.default_joint_armatures * avg_body_size_factor
        default_joint_stiffnesses = self.default_joint_stiffnesses * avg_body_size_factor
        default_joint_frictionlosses = self.default_joint_frictionlosses * avg_body_size_factor
        default_p_gain = self.default_p_gain * avg_body_size_factor
        default_d_gain = self.default_d_gain * avg_body_size_factor
        default_scaling_factor = self.default_scaling_factor * avg_body_size_factor

        coupled_masses = default_masses * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.coupled_mass_inertia_factor, high=self.coupled_mass_inertia_factor, size=self.default_masses.shape))
        coupled_inertias = default_inertias * (np.reshape(coupled_masses / default_masses, (-1, 1)))
        seen_body_masses = coupled_masses * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.decoupled_mass_inertia_factor, high=self.decoupled_mass_inertia_factor, size=coupled_masses.shape))
        seen_inertias = coupled_inertias * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.decoupled_mass_inertia_factor, high=self.decoupled_mass_inertia_factor, size=coupled_inertias.shape))
        masses = seen_body_masses * self.env.internal_state["mass_inertia_noise_factors"]
        inertias = seen_inertias * self.env.internal_state["mass_inertia_noise_factors"].reshape(-1, 1)

        seen_coms = default_coms + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_com_displacement, high=self.add_com_displacement, size=self.default_coms.shape)
        coms = seen_coms * self.env.internal_state["com_noise_factors"]

        axes = self.env.np_rng.normal(size=(self.default_inertia_quats.shape[0], 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        noisy_angles = env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_inertia_orientation_rad, high=self.add_inertia_orientation_rad, size=(self.default_inertia_quats.shape[0],))
        noisy_rotations = Rotation.from_rotvec(axes * noisy_angles.reshape(-1, 1))
        default_rotations = Rotation.from_quat(self.default_inertia_quats[:, [1, 2, 3, 0]])
        inertia_quats = (noisy_rotations * default_rotations).as_quat(scalar_first=True)

        seen_body_positions = default_body_positions + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_body_position, high=self.add_body_position, size=self.default_body_positions.shape)
        seen_body_positions = np.where(self.env.has_equality_constraints, default_body_positions, seen_body_positions)
        body_positions = seen_body_positions * self.env.internal_state["body_position_noise_factors"]

        axes = self.env.np_rng.normal(size=(self.default_body_quats.shape[0], 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        noisy_angles = env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_body_orientation_rad, high=self.add_body_orientation_rad, size=(self.default_body_quats.shape[0],))
        noisy_angles = np.where(self.env.has_equality_constraints, np.zeros_like(noisy_angles), noisy_angles)
        noisy_rotations = Rotation.from_rotvec(axes * noisy_angles.reshape(-1, 1))
        default_rotations = Rotation.from_quat(self.default_body_quats[:, [1, 2, 3, 0]])
        body_quats = (noisy_rotations * default_rotations).as_quat(scalar_first=True)

        site_positions = default_site_positions.copy()
        site_positions[self.env.imu_site_id] += env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_imu_position, high=self.add_imu_position, size=(3,))

        geom_sizes = default_geom_sizes
        geom_sizes[self.env.foot_geom_indices - 1] *= (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.foot_size_factor, high=self.foot_size_factor, size=(self.env.foot_geom_indices.shape[0], 3)))

        abs_axis = np.abs(self.default_joint_rotation_axes)
        min_axis = np.argmin(abs_axis, axis=1)
        perpendicular_vector = np.zeros_like(self.default_joint_rotation_axes)
        perpendicular_vector[np.arange(perpendicular_vector.shape[0]), (min_axis + 1) % 3] = self.default_joint_rotation_axes[np.arange(self.default_joint_rotation_axes.shape[0]), (min_axis + 2) % 3]
        perpendicular_vector[np.arange(perpendicular_vector.shape[0]), (min_axis + 2) % 3] = -self.default_joint_rotation_axes[np.arange(self.default_joint_rotation_axes.shape[0]), (min_axis + 1) % 3]
        perpendicular_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector, axis=1, keepdims=True)
        second_perpendicular_vector = np.cross(self.default_joint_rotation_axes, perpendicular_vector)
        phi = self.env.np_rng.uniform(low=0, high=2 * np.pi, size=(self.default_joint_rotation_axes.shape[0], 1))
        random_axis = np.cos(phi) * perpendicular_vector + np.sin(phi) * second_perpendicular_vector
        random_angle = env_curriculum_coeff * self.env.np_rng.uniform(low=0, high=self.joint_axis_angle_rad, size=(self.default_joint_rotation_axes.shape[0], 1))
        rotation = Rotation.from_rotvec(random_angle * random_axis)
        rotated_joint_rotation_axes = rotation.apply(self.default_joint_rotation_axes)

        torque_limits = default_torque_limits * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.torque_limit_factor, high=self.torque_limit_factor, size=self.default_torque_limits.shape))

        actuator_joint_nominal_positions = self.default_actuator_joint_nominal_positions + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_actuator_joint_nominal_position, high=self.add_actuator_joint_nominal_position, size=self.default_actuator_joint_nominal_positions.shape)
        actuator_joint_nominal_positions = np.clip(actuator_joint_nominal_positions, self.env.internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 0], self.env.internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 1])

        actuator_joint_max_velocities = default_actuator_joint_max_velocities * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.joint_velocity_max_factor, high=self.joint_velocity_max_factor, size=self.default_actuator_joint_max_velocities.shape))

        joint_ranges = self.default_joint_ranges + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_joint_range, high=self.add_joint_range, size=self.default_joint_ranges.shape)

        seen_joint_dampings = default_joint_dampings * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.joint_damping_factor, high=self.joint_damping_factor, size=self.default_joint_dampings.shape))
        seen_joint_dampings += env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_joint_damping, high=self.add_joint_damping, size=self.default_joint_dampings.shape)
        seen_joint_dampings = np.maximum(0, seen_joint_dampings)
        dof_damping = seen_joint_dampings * self.env.internal_state["joint_damping_noise_factors"]

        seen_joint_armatures = default_joint_armatures * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.joint_armature_factor, high=self.joint_armature_factor, size=self.default_joint_armatures.shape))
        seen_joint_armatures += env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_joint_armature, high=self.add_joint_armature, size=self.default_joint_armatures.shape)
        seen_joint_armatures = np.maximum(0, seen_joint_armatures)
        dof_armature = seen_joint_armatures * self.env.internal_state["joint_armature_noise_factors"]

        seen_joint_stiffnesses = default_joint_stiffnesses * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.joint_stiffness_factor, high=self.joint_stiffness_factor, size=self.default_joint_stiffnesses.shape))
        seen_joint_stiffnesses += env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_joint_stiffness, high=self.add_joint_stiffness, size=self.default_joint_stiffnesses.shape)
        seen_joint_stiffnesses = np.maximum(0, seen_joint_stiffnesses)
        jnt_stiffness = seen_joint_stiffnesses * self.env.internal_state["joint_stiffness_noise_factors"]

        seen_joint_frictionlosses = default_joint_frictionlosses * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.joint_friction_loss_factor, high=self.joint_friction_loss_factor, size=self.default_joint_frictionlosses.shape))
        seen_joint_frictionlosses += env_curriculum_coeff * self.env.np_rng.uniform(low=-self.add_joint_friction_loss, high=self.add_joint_friction_loss, size=self.default_joint_frictionlosses.shape)
        seen_joint_frictionlosses = np.maximum(0, seen_joint_frictionlosses)
        dof_frictionloss = seen_joint_frictionlosses * self.env.internal_state["joint_friction_loss_noise_factors"]

        seen_p_gain = default_p_gain * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.p_gain_factor, high=self.p_gain_factor))
        p_gain = seen_p_gain * self.env.internal_state["p_gain_noise_factors"]
        seen_d_gain = default_d_gain * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.d_gain_factor, high=self.d_gain_factor))
        d_gain = seen_d_gain * self.env.internal_state["d_gain_noise_factors"]
        scaling_factor = default_scaling_factor * (1 + env_curriculum_coeff * self.env.np_rng.uniform(low=-self.scaling_factor_factor, high=self.scaling_factor_factor))

        self.env.internal_state["mj_model"].body_pos[1:] = body_positions
        self.env.internal_state["mj_model"].body_quat[1:] = body_quats
        self.env.internal_state["mj_model"].body_mass[1:] = masses
        self.env.internal_state["mj_model"].body_inertia[1:] = inertias
        self.env.internal_state["mj_model"].body_ipos[1:] = coms
        self.env.internal_state["mj_model"].body_iquat[1:] = inertia_quats
        self.env.internal_state["mj_model"].site_pos = site_positions
        self.env.internal_state["mj_model"].geom_size[1:] = geom_sizes
        self.env.internal_state["mj_model"].geom_pos[1:] = geom_positions
        self.env.internal_state["mj_model"].geom_rbound[1:] = geom_rbounds
        self.env.internal_state["mj_model"].geom_aabb[1:] = geom_aabbs
        self.env.internal_state["mj_model"].mesh_pos = mesh_positions
        self.env.internal_state["mj_model"].mesh_vert = mesh_verts
        self.env.internal_state["mj_model"].site_size = site_sizes
        self.env.internal_state["mj_model"].cam_pos = camera_positions
        self.env.internal_state["mj_model"].jnt_pos[1:] = joint_positions
        self.env.internal_state["mj_model"].jnt_axis[1:] = rotated_joint_rotation_axes
        self.env.internal_state["mj_model"].actuator_forcerange[:, 1] = torque_limits
        self.env.internal_state["mj_model"].actuator_forcerange[:, 0] = -torque_limits
        self.env.internal_state["mj_model"].jnt_range[1:] = joint_ranges
        self.env.internal_state["mj_model"].dof_damping[6:] = dof_damping
        self.env.internal_state["mj_model"].dof_armature[6:] = dof_armature
        self.env.internal_state["mj_model"].jnt_stiffness[1:] = jnt_stiffness
        self.env.internal_state["mj_model"].dof_frictionloss[6:] = dof_frictionloss
        self.env.internal_state["mj_model"].actuator_gainprm[:, 0] = p_gain
        self.env.internal_state["mj_model"].actuator_biasprm[:, 1] = -p_gain
        self.env.internal_state["mj_model"].actuator_biasprm[:, 2] = -d_gain

        self.env.internal_state["seen_body_masses"] = seen_body_masses
        self.env.internal_state["seen_body_inertias"] = seen_inertias
        self.env.internal_state["seen_body_coms"] = seen_coms
        self.env.internal_state["seen_body_positions"] = seen_body_positions
        self.env.internal_state["actuator_joint_nominal_positions"] = actuator_joint_nominal_positions
        self.env.internal_state["actuator_joint_max_velocities"] = actuator_joint_max_velocities
        self.env.internal_state["seen_joint_ranges"] = joint_ranges
        self.env.internal_state["seen_joint_dampings"] = seen_joint_dampings
        self.env.internal_state["seen_joint_armatures"] = seen_joint_armatures
        self.env.internal_state["seen_joint_stiffnesses"] = seen_joint_stiffnesses
        self.env.internal_state["seen_joint_frictionlosses"] = seen_joint_frictionlosses
        self.env.internal_state["seen_p_gain"] = seen_p_gain
        self.env.internal_state["seen_d_gain"] = seen_d_gain
        self.env.internal_state["scaling_factor"] = scaling_factor
        self.env.internal_state["partial_actuator_gainprm_without_dropout"] = self.env.internal_state["mj_model"].actuator_gainprm[:, 0]
        self.env.internal_state["partial_actuator_biasprm_without_dropout"] = self.env.internal_state["mj_model"].actuator_biasprm[:, 1:3]

        qpos = self.env.initial_qpos.copy()
        qpos[self.env.actuator_joint_mask_qpos] = self.env.internal_state["actuator_joint_nominal_positions"]
        qpos[2] += self.env.internal_state["center_height"]
        qvel = np.zeros(self.env.initial_mj_model.nv)
        data = mujoco.MjData(self.env.internal_state["mj_model"])
        data.qpos = qpos
        data.qvel = qvel
        data.ctrl = np.zeros(self.env.nr_actuator_joints)
        mujoco.mj_forward(self.env.internal_state["mj_model"], data)
        min_feet_z_pos = np.min(data.geom_xpos[self.env.foot_geom_indices, 2])
        offset = self.env.internal_state["center_height"] - min_feet_z_pos
        robot_nominal_qpos_height_over_ground = qpos[2] - self.env.internal_state["center_height"] + offset
        robot_nominal_imu_height_over_ground = data.site_xpos[self.env.imu_site_id, 2] - self.env.internal_state["center_height"] + offset
        self.env.internal_state["robot_nominal_qpos_height_over_ground"] = robot_nominal_qpos_height_over_ground
        self.env.internal_state["robot_nominal_imu_height_over_ground"] = robot_nominal_imu_height_over_ground
        all_contact_relevant_geom_xpos = data.geom_xpos[self.env.reward_collision_sphere_geom_ids]
        all_contact_relevant_geom_sizes = self.env.internal_state["mj_model"].geom_size[self.env.reward_collision_sphere_geom_ids, 0]
        distance_between_geoms = np.linalg.norm(all_contact_relevant_geom_xpos[:, None] - all_contact_relevant_geom_xpos[None], axis=-1)
        contact_between_geoms = distance_between_geoms <= (all_contact_relevant_geom_sizes[:, None] + all_contact_relevant_geom_sizes[None])
        nr_collisions = (np.sum(contact_between_geoms) - len(self.env.reward_collision_sphere_geom_ids)) // 2
        self.env.internal_state["nr_collisions_in_nominal"] = nr_collisions

        data.qpos = self.env.internal_state["data"].qpos
        mujoco.mj_forward(self.env.internal_state["mj_model"], data)
        feet_x_pos = data.geom_xpos[self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = np.max(self.env.terrain_function.ground_height_at(feet_x_pos, feet_y_pos) - data.geom_xpos[self.env.foot_geom_indices, 2])
        self.env.internal_state["data"].qpos[2] += min_feet_z_pos_under_ground
