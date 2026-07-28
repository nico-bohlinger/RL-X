import torch
import torch.nn as nn


class BatchRenorm(nn.Module):
    def __init__(self, input_dim, momentum, warmup_steps):
        super().__init__()
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.scale = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))


    def forward(self, x, train):
        if not train:
            return (x - self.running_mean) * torch.rsqrt(self.running_var + 1e-3) * self.scale + self.bias

        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        with torch.no_grad():
            std = torch.sqrt(var + 1e-3)
            running_std = torch.sqrt(self.running_var + 1e-3)
            r = torch.clamp(std / running_std, 1.0 / 3.0, 3.0)
            d = torch.clamp((mean - self.running_mean) / running_std, -5.0, 5.0)
            warmed_up = (self.steps >= self.warmup_steps).float()
        custom_var = warmed_up * var / r ** 2 + (1.0 - warmed_up) * var
        custom_mean = warmed_up * (mean - d * torch.sqrt(var) / r) + (1.0 - warmed_up) * mean
        self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean.detach()
        self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * var.detach()
        self.steps = self.steps + 1
        return (x - custom_mean) * torch.rsqrt(custom_var + 1e-3) * self.scale + self.bias
