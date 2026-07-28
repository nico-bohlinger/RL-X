import numpy as np
import torch
import torch.nn as nn


class PreLNResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden_1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.hidden_2 = nn.Linear(hidden_dim * 4, hidden_dim)
        std = np.sqrt(2.0 / hidden_dim) / 0.87962566103423978
        nn.init.trunc_normal_(self.hidden_1.weight, std=std, a=-2.0 * std, b=2.0 * std)
        nn.init.zeros_(self.hidden_1.bias)
        std = np.sqrt(2.0 / (hidden_dim * 4)) / 0.87962566103423978
        nn.init.trunc_normal_(self.hidden_2.weight, std=std, a=-2.0 * std, b=2.0 * std)
        nn.init.zeros_(self.hidden_2.bias)


    def forward(self, x):
        residual = x
        x = self.hidden_2(torch.relu(self.hidden_1(self.layer_norm(x))))
        return residual + x


class SimbaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nr_blocks):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([PreLNResidualBlock(hidden_dim) for _ in range(nr_blocks)])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        nn.init.orthogonal_(self.input.weight, 1.0)
        nn.init.zeros_(self.input.bias)


    def forward(self, x):
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.layer_norm(x)
