import jax.numpy as jnp
from mujoco import mjx


class DefaultDRInitialState:
    def __init__(self, env):
        self.env = env


    def setup(self, mjx_model, internal_state, key):
        nr_envs = self.env.nr_envs
        qpos = jnp.tile(self.env.initial_qpos[None], (nr_envs, 1))
        qpos = qpos.at[:, self.env.actuator_joint_mask_qpos].set(internal_state["actuator_joint_nominal_positions"])
        qpos = qpos.at[:, 2].set(internal_state["robot_nominal_qpos_height_over_ground"] + internal_state["center_height"])

        qvel = jnp.zeros((nr_envs, self.env.initial_mj_model.nv))

        data = self.env.mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((nr_envs, self.env.nr_actuator_joints)))
        data = mjx.forward(mjx_model, data)
        feet_x_pos = data.geom_xpos[:, self.env.foot_geom_indices, 0]
        feet_y_pos = data.geom_xpos[:, self.env.foot_geom_indices, 1]
        min_feet_z_pos_under_ground = jnp.max(self.env.terrain_function.ground_height_at(internal_state, feet_x_pos, feet_y_pos) - data.geom_xpos[:, self.env.foot_geom_indices, 2], axis=-1)
        qpos = qpos.at[:, 2].set(qpos[:, 2] + min_feet_z_pos_under_ground)

        return qpos, qvel
