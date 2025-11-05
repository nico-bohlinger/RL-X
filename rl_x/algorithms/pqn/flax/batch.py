class Batch:
    def __init__(self, states, next_states, actions, rewards, terminations, q_targets):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.terminations = terminations
        self.q_targets = q_targets
