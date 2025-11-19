class NoneActionDelay:
    def __init__(self, env):
        self.env = env


    def init(self):
        pass


    def setup(self):
        pass


    def sample(self, should_randomize):
        pass


    def delay_action(self, action):
        return action
