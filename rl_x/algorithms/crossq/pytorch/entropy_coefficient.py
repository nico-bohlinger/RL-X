import numpy as np
import torch
import torch.nn as nn


def get_entropy_coefficient(config, env, device):
    if config.algorithm.target_entropy == "auto":
        target_entropy = -np.prod(env.single_action_space.shape, dtype=int).item()
    else:
        target_entropy = float(config.algorithm.target_entropy)
    return EntropyCoefficient(target_entropy, device).to(device)


class EntropyCoefficient(nn.Module):
    def __init__(self, target_entropy, device):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
        self.target_entropy = target_entropy


    def forward(self):
        return torch.exp(self.log_alpha)


    def loss(self, entropy):
        return self() * (entropy - self.target_entropy)
