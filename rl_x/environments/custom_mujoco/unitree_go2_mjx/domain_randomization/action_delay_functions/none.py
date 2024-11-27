class NoneActionDelay:
    def __init__(self, env):
        self.env = env


    def setup(self, internal_state):
        pass


    def sample(self, internal_state, should_randomize, key):
        pass


    def delay_action(self, action, internal_state, key):
        return action
