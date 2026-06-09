import jax
import jax.numpy as jnp


class StepProbabilityAndResetSampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self, subkey, curriculum_coeff=1.0):
        return jnp.ones(self.env.nr_envs, dtype=bool)


    def step(self, subkey, curriculum_coeff=1.0):
        return jax.random.uniform(subkey, (self.env.nr_envs,)) < self.probability * curriculum_coeff
