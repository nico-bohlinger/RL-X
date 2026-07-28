class Batch:
    def __init__(self, states, next_states, actions, rewards, values, terminations, dones, log_probs, advantages, returns, init_policy_carry):
        self.states = states
        self.next_states = next_states
        self.actions = actions
        self.rewards = rewards
        self.values = values
        self.terminations = terminations
        self.dones = dones
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns
        self.init_policy_carry = init_policy_carry
