import math
import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, dim, init, scale):
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), scale))
        self.factor = init / scale


    def forward(self, x):
        return self.scaler.to(x.dtype) * self.factor * x


class HyperDense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.linear.weight, 1.0)


    def forward(self, x):
        return self.linear(x)


class HyperEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, scaler_init, scaler_scale, c_shift):
        super().__init__()
        self.c_shift = c_shift
        self.dense = HyperDense(input_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)


    def forward(self, x):
        shift_axis = torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device) * self.c_shift
        x = torch.nn.functional.normalize(torch.cat([x, shift_axis], dim=-1), dim=-1, eps=1e-8)
        return torch.nn.functional.normalize(self.scaler(self.dense(x)), dim=-1, eps=1e-8)


class HyperMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, scaler_init, scaler_scale):
        super().__init__()
        self.hidden = HyperDense(output_dim, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.output = HyperDense(hidden_dim, output_dim)


    def forward(self, x):
        x = torch.relu(self.scaler(self.hidden(x))) + 1e-8
        return torch.nn.functional.normalize(self.output(x), dim=-1, eps=1e-8)


class HyperLERPBlock(nn.Module):
    def __init__(self, hidden_dim, scaler_init, scaler_scale, alpha_init, alpha_scale):
        super().__init__()
        self.mlp = HyperMLP(hidden_dim * 4, hidden_dim, scaler_init / math.sqrt(4), scaler_scale / math.sqrt(4))
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)


    def forward(self, x):
        residual = x
        x = residual + self.alpha(self.mlp(x) - residual)
        return torch.nn.functional.normalize(x, dim=-1, eps=1e-8)


def l2normalize_parameters(module):
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, HyperDense):
                child.linear.weight.div_(torch.clamp(torch.linalg.vector_norm(child.linear.weight, dim=1, keepdim=True), min=1e-8))
