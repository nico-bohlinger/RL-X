import math
import torch
import torch.nn as nn


class EntropyCoefficient(nn.Module):
    def __init__(self, initial_value):
        super().__init__()
        self.log_value = nn.Parameter(torch.tensor(math.log(initial_value), dtype=torch.float32))


    def forward(self):
        return torch.exp(self.log_value)


class Adjustment(nn.Module):
    def __init__(self, initial_value, log_value_min, log_value_max):
        super().__init__()
        ratio = (math.log(initial_value) - log_value_min) / ((log_value_max - log_value_min) * 0.5) - 1.0
        self.raw_value = nn.Parameter(torch.tensor(math.atanh(ratio), dtype=torch.float32))
        self.log_value_min = log_value_min
        self.log_value_max = log_value_max


    def forward(self):
        log_value = self.log_value_min + (self.log_value_max - self.log_value_min) * 0.5 * (1.0 + torch.tanh(self.raw_value))
        return torch.exp(log_value)
