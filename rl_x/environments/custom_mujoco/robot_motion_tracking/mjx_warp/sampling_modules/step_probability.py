import jax
import jax.numpy as jnp


class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self, is_initial=False):
        return jnp.zeros(self.env.nr_envs, dtype=bool)


    def step(self, key):
        return jax.random.uniform(key, (self.env.nr_envs,)) < self.probability
