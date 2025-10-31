import jax.numpy as jnp


class DefaultDRInitialState:
    def __init__(self, env):
        self.env = env

        self.initial_qpos = jnp.array(self.env.initial_mj_model.keyframe("home").qpos)
        self.initial_qvel = jnp.zeros(self.env.initial_mjx_model.nv)


    def setup(self, data, mjx_model, internal_state, key):
        qpos = self.initial_qpos
        qpos = qpos.at[2].set(qpos[2] + internal_state["center_height"])

        qvel = self.initial_qvel
        
        return qpos, qvel
