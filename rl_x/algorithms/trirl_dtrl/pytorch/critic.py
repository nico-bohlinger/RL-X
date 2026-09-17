import torch
from torch import nn
from rl_x.algorithms.trirl_dtrl.pytorch.network import Network


class Critic(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        indices = getattr(env, "critic_observation_indices", range(env.single_observation_space.shape[0]))
        self.register_buffer("observation_indices", torch.as_tensor(list(indices), dtype=torch.long))
        self.network = Network(len(indices), [512, 256, 128, 1], 1.0, layer_norm=True)


    def forward(self, x):
        return self.network(x[..., self.observation_indices])
