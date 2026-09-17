from torch import nn
from rl_x.algorithms.airl_ppo.pytorch.network import Network


class Discriminator(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.gamma = config.algorithm.gamma
        self.gnet = Network(env.single_observation_space.shape[0], [512, 256, 1], 0.1, activation="tanh")
        self.hnet = Network(env.single_observation_space.shape[0], [512, 256, 1], 0.1, activation="tanh")


    def forward(self, state, action, next_state, absorbing, log_prob, shaping=1.0):
        reward = self.gnet(state).squeeze(-1)
        value = self.hnet(state).squeeze(-1)
        next_value = self.hnet(next_state).squeeze(-1)
        if self.handle_absorbing_states:
            next_reward = self.gnet(next_state).squeeze(-1)
            potential = (1 - absorbing) * self.gamma * next_value + absorbing * self.gamma / (1 - self.gamma) * next_reward - value
        else:
            potential = self.gamma * next_value - value
        return reward + shaping * (potential - log_prob)
