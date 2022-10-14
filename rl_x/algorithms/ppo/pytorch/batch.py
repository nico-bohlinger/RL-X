class Batch:
    def __init__(self, states, actions, rewards, returns, values, advantages, dones, log_probs):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.returns = returns
        self.values = values
        self.advantages = advantages
        self.dones = dones
        self.log_probs = log_probs
