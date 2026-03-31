import jax
import jax.numpy as jnp
from mujoco import mjx


class DefaultDRInitialState:
    def __init__(self, env):
        self.env = env


    def setup(self, mjx_model, internal_state, key):
        qpos = self.env.initial_qpos.at[self.env.actuator_joint_mask_qpos].set(internal_state["actuator_joint_nominal_positions"])
        qpos = qpos.at[2].set(internal_state["robot_nominal_qpos_height_over_ground"] + internal_state["center_height"])

        qvel = jnp.zeros(self.env.initial_mjx_model.nv)

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
