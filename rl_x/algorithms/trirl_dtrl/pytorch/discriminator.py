import torch
from torch import nn
from rl_x.algorithms.trirl_dtrl.pytorch.network import Network


class Discriminator(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.handle_absorbing_states = config.algorithm.handle_absorbing_states
        self.state_based = config.algorithm.get("reward_type", "state-action") == "state-based"
        size = env.single_observation_space.shape[0] + (env.single_observation_space.shape[0] if self.state_based else env.single_action_space.shape[0]) + int(self.handle_absorbing_states)
        self.network = Network(size, [512, 256, 1], 0.1, activation="tanh")


    def forward(self, state, action, next_state, absorbing, log_prob=None):
        parts = [state, absorbing[..., None], next_state if self.state_based else action] if self.handle_absorbing_states else [state, next_state if self.state_based else action]
        return self.network(torch.cat(parts, dim=-1)).squeeze(-1)
