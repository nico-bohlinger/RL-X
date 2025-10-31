import jax


class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self, subkey):
        return False


    def step(self, subkey):
        return jax.random.uniform(subkey) < self.probability
