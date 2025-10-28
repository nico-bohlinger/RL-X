class Batch:
    def __init__(self, states, next_states, actions, rewards, values, terminations, log_probs, advantages, returns, old_action_means, old_action_logstd):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns
        self.old_action_means = old_action_means
        self.old_action_logstd = old_action_logstd
