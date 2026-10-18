import torch
from torch import nn
from rl_x.algorithms.trirl_ppo.pytorch.network import Network


class RewardFunction(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        self.network = Network(env.single_observation_space.shape[0] + env.single_action_space.shape[0], [512, 256, 1], 1.0, activation="relu")


    def forward(self, state, action, next_state=None):
        return self.network(torch.cat([state, action], dim=-1)).squeeze(-1)
