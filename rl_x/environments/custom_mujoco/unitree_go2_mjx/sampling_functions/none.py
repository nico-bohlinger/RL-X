class NoneSampling:
    def __init__(self, env):
        self.env = env


    def setup(self, subkey):
        return False


    def step(self, subkey):
        return False
