class NoneDRJointDropout:
    def __init__(self, env):
        self.env = env


    def init(self, internal_state):
        pass


    def sample(self, internal_state, mjx_model, should_randomize, key):
        return mjx_model
