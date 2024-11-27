class NoneDRObservationDropout:
    def __init__(self, env):
        self.env = env


    def modify_observation(self, observation, key):
        return observation
