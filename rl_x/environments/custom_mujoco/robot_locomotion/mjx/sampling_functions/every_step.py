class EveryStepSampling:
    def __init__(self, env):
        self.env = env


    def setup(self, subkey, curriculum_coeff=1.0):
        return True


    def step(self, subkey, curriculum_coeff=1.0):
        return True
