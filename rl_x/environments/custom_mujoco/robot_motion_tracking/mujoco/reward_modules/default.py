import numpy as np


class DefaultReward:
    def __init__(self, env):
        self.env = env

        self.anchor_position_coeff = env.env_config["reward"]["default"]["anchor_position_coeff"] * env.dt
        self.anchor_position_sigma = env.env_config["reward"]["default"]["anchor_position_sigma"]
        self.anchor_orientation_coeff = env.env_config["reward"]["default"]["anchor_orientation_coeff"] * env.dt
        self.anchor_orientation_sigma = env.env_config["reward"]["default"]["anchor_orientation_sigma"]
        self.body_position_coeff = env.env_config["reward"]["default"]["body_position_coeff"] * env.dt
        self.body_position_sigma = env.env_config["reward"]["default"]["body_position_sigma"]
        self.body_orientation_coeff = env.env_config["reward"]["default"]["body_orientation_coeff"] * env.dt
        self.body_orientation_sigma = env.env_config["reward"]["default"]["body_orientation_sigma"]
        self.body_linear_velocity_coeff = env.env_config["reward"]["default"]["body_linear_velocity_coeff"] * env.dt
        self.body_linear_velocity_sigma = env.env_config["reward"]["default"]["body_linear_velocity_sigma"]
        self.body_angular_velocity_coeff = env.env_config["reward"]["default"]["body_angular_velocity_coeff"] * env.dt
        self.body_angular_velocity_sigma = env.env_config["reward"]["default"]["body_angular_velocity_sigma"]
        self.object_position_coeff = env.env_config["reward"]["default"]["object_position_coeff"] * env.dt
        self.object_position_sigma = env.env_config["reward"]["default"]["object_position_sigma"]
        self.object_orientation_coeff = env.env_config["reward"]["default"]["object_orientation_coeff"] * env.dt
        self.object_orientation_sigma = env.env_config["reward"]["default"]["object_orientation_sigma"]
        self.action_rate_coeff = env.env_config["reward"]["default"]["action_rate_coeff"] * env.dt
        self.joint_limits_coeff = env.env_config["reward"]["default"]["joint_limits_coeff"] * env.dt

        self.undesired_contacts_coeff = env.env_config["reward"]["default"]["undesired_contacts_coeff"] * env.dt
        self.contact_force_threshold = env.env_config["reward"]["default"]["contact_force_threshold"]
        self.contact_history_length = env.env_config["reward"]["default"]["contact_history_length"]
        self.undesired_contact_body_ids = np.asarray([body for body in range(env.initial_mj_model.nbody) if env.initial_mj_model.body_rootid[body] == env.trunk_body_id and env.initial_mj_model.body(body).name not in env.env_config["reward"]["default"]["allowed_contact_names"]], dtype=int)
        soft_limit = env.env_config["reward"]["default"]["soft_joint_position_limit"]
        self.soft_lower = env.lower + (1 - soft_limit) / 2 * (env.upper - env.lower)
        self.soft_upper = env.upper - (1 - soft_limit) / 2 * (env.upper - env.lower)


    def init(self):
        self.setup()


    def setup(self):
        self.env.internal_state["contact_forces"] = np.zeros((self.contact_history_length, self.env.initial_mj_model.nbody, 3), dtype=np.float32)


    def step(self):
        self.env.internal_state["contact_forces"] = np.concatenate((self.env.internal_state["data"].cfrc_ext[None, :, self.env.body_linear_velocity_slice], self.env.internal_state["contact_forces"][:-1]), axis=0)


    def reward_and_info(self, action):
        # Anchor position reward
        anchor_position_difference = self.env.internal_state["reference"]["anchor_position"] - self.env.internal_state["anchor_position"]
        anchor_position_error_squared = np.sum(np.square(anchor_position_difference), axis=-1)
        anchor_position_reward = self.anchor_position_coeff * np.exp(-anchor_position_error_squared / self.anchor_position_sigma**2)

        # Anchor orientation reward
        anchor_orientation_error_squared = np.square((self.env.internal_state["reference"]["anchor_rotation"] * self.env.internal_state["anchor_rotation_inverse"]).magnitude())
        anchor_orientation_reward = self.anchor_orientation_coeff * np.exp(-anchor_orientation_error_squared / self.anchor_orientation_sigma**2)

        # Body position reward
        body_position_difference = self.env.internal_state["reference"]["aligned_body_positions"] - self.env.internal_state["body_positions"]
        body_position_error_squared = np.mean(np.sum(np.square(body_position_difference), axis=-1), axis=-1)
        body_position_reward = self.body_position_coeff * np.exp(-body_position_error_squared / self.body_position_sigma**2)

        # Body orientation reward
        body_orientation_error_squared = np.mean(np.square((self.env.internal_state["reference"]["aligned_body_rotation"] * self.env.internal_state["body_rotation"].inv()).magnitude().reshape(self.env.internal_state["body_positions"].shape[:-1])), axis=-1)
        body_orientation_reward = self.body_orientation_coeff * np.exp(-body_orientation_error_squared / self.body_orientation_sigma**2)

        # Body linear velocity reward
        body_linear_velocity_difference = self.env.internal_state["reference"]["body_velocities"][..., self.env.body_linear_velocity_slice] - self.env.internal_state["body_velocities"][..., self.env.body_linear_velocity_slice]
        body_linear_velocity_error_squared = np.mean(np.sum(np.square(body_linear_velocity_difference), axis=-1), axis=-1)
        body_linear_velocity_reward = self.body_linear_velocity_coeff * np.exp(-body_linear_velocity_error_squared / self.body_linear_velocity_sigma**2)

        # Body angular velocity reward
        body_angular_velocity_difference = self.env.internal_state["reference"]["body_velocities"][..., self.env.body_angular_velocity_slice] - self.env.internal_state["body_velocities"][..., self.env.body_angular_velocity_slice]
        body_angular_velocity_error_squared = np.mean(np.sum(np.square(body_angular_velocity_difference), axis=-1), axis=-1)
        body_angular_velocity_reward = self.body_angular_velocity_coeff * np.exp(-body_angular_velocity_error_squared / self.body_angular_velocity_sigma**2)

        # Object pose rewards
        if self.env.motion_library.has_object:
            object_position_difference = self.env.internal_state["reference"]["qpos"][..., self.env.object_position_qpos_slice] - self.env.internal_state["data"].qpos[..., self.env.object_position_qpos_slice]
            object_position_error_squared = np.sum(np.square(object_position_difference), axis=-1)
            object_position_reward = self.object_position_coeff * np.exp(-object_position_error_squared / self.object_position_sigma**2)

            object_orientation_error_squared = np.square((self.env.internal_state["reference"]["object_rotation"] * self.env.internal_state["object_rotation"].inv()).magnitude())
            object_orientation_reward = self.object_orientation_coeff * np.exp(-object_orientation_error_squared / self.object_orientation_sigma**2)

        # Action rate penalty
        action_difference = action - self.env.internal_state["last_action"]
        action_rate_penalty = np.sum(np.square(action_difference), axis=-1)
        action_rate_reward = self.action_rate_coeff * action_rate_penalty

        # Joint position limit penalty
        lower_limit_penalty = np.maximum(self.soft_lower - self.env.internal_state["data"].qpos[..., self.env.actuator_joint_mask_qpos], 0)
        upper_limit_penalty = np.maximum(self.env.internal_state["data"].qpos[..., self.env.actuator_joint_mask_qpos] - self.soft_upper, 0)
        joint_limits_penalty = np.sum(lower_limit_penalty + upper_limit_penalty, axis=-1)
        joint_limits_reward = self.joint_limits_coeff * joint_limits_penalty

        # Undesired contact penalty
        maximum_contact_force = np.max(np.linalg.norm(self.env.internal_state["contact_forces"][..., self.undesired_contact_body_ids, :], axis=-1), axis=-2)
        undesired_contacts_penalty = np.sum(maximum_contact_force > self.contact_force_threshold, axis=-1)
        undesired_contacts_reward = self.undesired_contacts_coeff * undesired_contacts_penalty

        # Total reward
        reward = anchor_position_reward + anchor_orientation_reward + body_position_reward + body_orientation_reward + body_linear_velocity_reward + body_angular_velocity_reward
        if self.env.motion_library.has_object:
            reward = reward + object_position_reward + object_orientation_reward
        reward = reward + action_rate_reward + joint_limits_reward + undesired_contacts_reward

        # Info
        self.env.internal_state["info"]["reward/anchor_position"] = anchor_position_reward
        self.env.internal_state["info"]["reward/anchor_orientation"] = anchor_orientation_reward
        self.env.internal_state["info"]["reward/body_position"] = body_position_reward
        self.env.internal_state["info"]["reward/body_orientation"] = body_orientation_reward
        self.env.internal_state["info"]["reward/body_linear_velocity"] = body_linear_velocity_reward
        self.env.internal_state["info"]["reward/body_angular_velocity"] = body_angular_velocity_reward
        self.env.internal_state["info"]["reward/action_rate"] = action_rate_reward
        self.env.internal_state["info"]["reward/joint_limits"] = joint_limits_reward
        self.env.internal_state["info"]["reward/undesired_contacts"] = undesired_contacts_reward
        self.env.internal_state["info"]["reward/total"] = reward
        self.env.internal_state["info"]["tracking/anchor_position_error"] = np.sqrt(anchor_position_error_squared)
        self.env.internal_state["info"]["tracking/anchor_orientation_error"] = np.sqrt(anchor_orientation_error_squared)
        self.env.internal_state["info"]["tracking/body_position_error"] = np.sqrt(body_position_error_squared)
        self.env.internal_state["info"]["tracking/body_orientation_error"] = np.sqrt(body_orientation_error_squared)
        self.env.internal_state["info"]["tracking/body_linear_velocity_error"] = np.sqrt(body_linear_velocity_error_squared)
        self.env.internal_state["info"]["tracking/body_angular_velocity_error"] = np.sqrt(body_angular_velocity_error_squared)
        if self.env.motion_library.has_object:
            self.env.internal_state["info"]["reward/object_position"] = object_position_reward
            self.env.internal_state["info"]["reward/object_orientation"] = object_orientation_reward
            self.env.internal_state["info"]["tracking/object_position_error"] = np.sqrt(object_position_error_squared)
            self.env.internal_state["info"]["tracking/object_orientation_error"] = np.sqrt(object_orientation_error_squared)

        return reward
