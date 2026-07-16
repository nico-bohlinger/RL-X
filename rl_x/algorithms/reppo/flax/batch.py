class Batch:
    def __init__(self, states, actions, rewards, soft_rewards, next_features, next_values, terminations, truncations):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.soft_rewards = soft_rewards
        self.next_features = next_features
        self.next_values = next_values
        self.terminations = terminations
        self.truncations = truncations
