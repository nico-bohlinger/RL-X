import numpy as np


class NoneActionDelay:
    def __init__(self, env):
        self.env = env


    def init(self):
        pass


    def setup(self):
        pass


    def sample(self):
        pass


    def delay_action(self, action):
        return np.broadcast_to(action, (self.env.nr_substeps, action.shape[0]))
