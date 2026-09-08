import numpy as np
from scipy.spatial.transform import Rotation


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


    def sample(self, qpos, qvel):
        qpos, qvel = qpos.copy(), qvel.copy()
        if self.enable_randomization and not self.env.internal_state["in_eval_mode"]:
            qpos[self.env.actuator_joint_mask_qpos] += self.env.np_rng.uniform(-self.joint_position, self.joint_position, self.env.nr_actuators)
            qpos[self.env.root_position_qpos_slice] += self.env.np_rng.uniform(-1.0, 1.0, 3) * self.root_position

            rotation = Rotation.from_quat(qpos[self.env.root_quaternion_qpos_slice][self.env.quaternion_wxyz_to_xyzw_indices])
            angular_velocity = rotation.apply(qvel[self.env.root_angular_qvel_slice])
            angles = self.env.np_rng.uniform(-1.0, 1.0, 3) * self.root_orientation
            rotation = Rotation.from_euler("xyz", angles) * rotation
            qpos[self.env.root_quaternion_qpos_slice] = rotation.as_quat()[self.env.quaternion_xyzw_to_wxyz_indices]

            qvel[self.env.root_linear_qvel_slice] += self.env.np_rng.uniform(-1.0, 1.0, 3) * self.root_linear_velocity
            angular_velocity += self.env.np_rng.uniform(-1.0, 1.0, 3) * self.root_angular_velocity
            qvel[self.env.root_angular_qvel_slice] = rotation.inv().apply(angular_velocity)

            if self.env.has_object:
                qpos[self.env.object_position_qpos_slice] += self.env.np_rng.uniform(-1.0, 1.0, 3) * self.object_position

        qpos[self.env.actuator_joint_mask_qpos] = np.clip(qpos[self.env.actuator_joint_mask_qpos], self.lower, self.upper)

        if self.env.has_object:
            qvel[self.env.object_angular_qvel_slice] = 0.0

        return qpos, qvel
