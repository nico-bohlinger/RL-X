import jax


class StepProbabilitySampling:
    def __init__(self, env, probability=0.002):
        self.env = env
        self.probability = probability


    def setup(self, subkey, curriculum_coeff=1.0):
        return False


    def step(self, subkey, curriculum_coeff=1.0):
        return jax.random.uniform(subkey) < self.probability * curriculum_coeff
