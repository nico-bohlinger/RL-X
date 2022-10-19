class Batch:
    def __init__(self, states, actions, rewards, values, dones, log_probs):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.dones = dones
        self.log_probs = log_probs
