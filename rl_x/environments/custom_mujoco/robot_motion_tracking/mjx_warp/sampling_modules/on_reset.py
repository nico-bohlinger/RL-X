import jax.numpy as jnp


class OnResetSampling:
    def __init__(self, env):
        self.env = env


    def setup(self, is_initial=False):
        return jnp.full(self.env.nr_envs, not is_initial, dtype=bool)


    def step(self, key):
        return jnp.zeros(self.env.nr_envs, dtype=bool)
