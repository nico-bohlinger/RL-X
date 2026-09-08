class OnCreationSampling:
    def __init__(self, env):
        self.env = env


    def setup(self, is_initial=False):
        return is_initial


    def step(self):
        return False
