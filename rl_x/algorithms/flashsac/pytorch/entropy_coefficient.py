import math
import torch
import torch.nn as nn


def get_entropy_coefficient(config, device):
    entropy_coefficient = EntropyCoefficient(config.algorithm.init_entropy_coefficient).to(device)
    if config.algorithm.use_compile:
        entropy_coefficient.forward = torch.compile(entropy_coefficient.forward, mode=config.algorithm.compile_mode)
    return entropy_coefficient


class EntropyCoefficient(nn.Module):
    def __init__(self, initial_value=0.01):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))


    def forward(self):
        return torch.exp(self.log_alpha)
