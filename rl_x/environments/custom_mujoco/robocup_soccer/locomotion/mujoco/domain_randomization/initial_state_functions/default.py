import numpy as np
import mujoco


class DefaultDRInitialState:
    def __init__(self, env):
        self.env = env


    def setup(self):
        qpos = self.env.initial_qpos.copy()
        qpos[self.env.actuator_joint_mask_qpos] = self.env.internal_state["actuator_joint_nominal_positions"]
        qpos[2] = self.env.internal_state["robot_nominal_qpos_height_over_ground"] + self.env.internal_state["center_height"]

        qvel = np.zeros(self.env.initial_mj_model.nv)

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
