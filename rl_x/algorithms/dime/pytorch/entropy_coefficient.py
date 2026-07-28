import numpy as np
import torch
import torch.nn as nn


def get_entropy_coefficient(config, device):
    return EntropyCoefficient(config.algorithm.entropy_coefficient_init).to(device)


class EntropyCoefficient(nn.Module):
    def __init__(self, initial_value):
        super().__init__()
        self.log_coefficient = nn.Parameter(torch.tensor(np.log(initial_value), dtype=torch.float32))


    def forward(self):
        return torch.exp(self.log_coefficient)
