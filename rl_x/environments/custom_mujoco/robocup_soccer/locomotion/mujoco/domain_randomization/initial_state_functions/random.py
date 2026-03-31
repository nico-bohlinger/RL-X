import numpy as np
from scipy.spatial.transform import Rotation
import mujoco


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


    def setup(self):
        roll_angle = self.env.np_rng.uniform(low=-np.pi * self.roll_angle_pi_factor, high=np.pi * self.roll_angle_pi_factor)
        pitch_angle = self.env.np_rng.uniform(low=-np.pi * self.pitch_angle_pi_factor, high=np.pi * self.pitch_angle_pi_factor)
        yaw_angle = self.env.np_rng.uniform(low=-np.pi * self.yaw_angle_pi_factor, high=np.pi * self.yaw_angle_pi_factor)
        quaternion = Rotation.from_euler("xyz", self.env.internal_state["env_curriculum_coeff"] * np.array([roll_angle, pitch_angle, yaw_angle])).as_quat(scalar_first=True)
        
        actuator_joint_nominal_position_factor = self.env.internal_state["env_curriculum_coeff"] * self.actuator_joint_nominal_position_factor
        actuator_joint_positions = self.env.internal_state["actuator_joint_nominal_positions"] * self.env.np_rng.uniform(low=1 - actuator_joint_nominal_position_factor, high=1 + actuator_joint_nominal_position_factor, size=self.env.internal_state["actuator_joint_nominal_positions"].size)
        actuator_joint_positions += self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.actuator_joint_position_offset_to_nominal, high=self.actuator_joint_position_offset_to_nominal, size=self.env.internal_state["actuator_joint_nominal_positions"].size)
        actuator_joint_positions = np.clip(actuator_joint_positions, self.env.internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 0], self.env.internal_state["joint_position_limits"][self.env.actuator_joint_mask_joints - 1, 1])

        joint_velocity_max_factor = self.env.internal_state["env_curriculum_coeff"] * self.joint_velocity_max_factor
        actuator_joint_velocities = self.env.internal_state["actuator_joint_max_velocities"] * self.env.np_rng.uniform(low=-joint_velocity_max_factor, high=joint_velocity_max_factor, size=self.env.actuator_joint_max_velocities.size)
        
        max_trunk_velocity = np.minimum(np.sum(self.env.internal_state["mj_model"].body_mass) * self.trunk_velocity_clip_mass_factor, self.trunk_velocity_clip_limit)
        linear_velocities = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-max_trunk_velocity, high=max_trunk_velocity, size=(3,))
        angular_velocities = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-max_trunk_velocity, high=max_trunk_velocity, size=(3,))
        
        linear_positions = np.array([0.0, 0.0, self.env.internal_state["robot_nominal_qpos_height_over_ground"] + self.env.internal_state["center_height"]])

        qpos = self.env.initial_qpos.copy()
        qpos[:3] = linear_positions
        qpos[3:7] = quaternion
        qpos[self.env.actuator_joint_mask_qpos] = actuator_joint_positions

        qvel = np.zeros(self.env.initial_mj_model.nv)
        qvel[:3] = linear_velocities
        qvel[3:6] = angular_velocities
        qvel[self.env.actuator_joint_mask_qvel] = actuator_joint_velocities

        data = mujoco.MjData(self.env.internal_state["mj_model"])
        data.qpos = qpos
        data.qvel = qvel
        data.ctrl = np.zeros(self.env.nr_actuator_joints)
        mujoco.mj_forward(self.env.internal_state["mj_model"], data)
        feet_x_pos = data.geom_xpos[self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = np.max(self.env.terrain_function.ground_height_at(feet_x_pos, feet_y_pos) - data.geom_xpos[self.env.foot_geom_indices, 2])
        qpos[2] += min_feet_z_pos_under_ground

        return qpos, qvel
