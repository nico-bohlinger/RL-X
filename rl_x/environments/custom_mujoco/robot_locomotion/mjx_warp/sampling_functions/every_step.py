import jax.numpy as jnp


class EveryStepSampling:
    def __init__(self, env):
        self.env = env


    def setup(self, subkey, curriculum_coeff=1.0):
        return jnp.ones(self.env.nr_envs, dtype=bool)


    def step(self, subkey, curriculum_coeff=1.0):
        return jnp.ones(self.env.nr_envs, dtype=bool)
