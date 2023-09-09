class Batch:
    def __init__(self, states, next_states, actions, rewards, values, terminations, log_probs):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
