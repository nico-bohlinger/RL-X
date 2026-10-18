import math
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self, input_size, widths, output_gain, activation="elu", layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = activation
        for index, width in enumerate(widths):
            layer = nn.Linear(input_size, width)
            nn.init.orthogonal_(layer.weight, output_gain if index == len(widths) - 1 else math.sqrt(2))
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(width, eps=1e-6) if layer_norm and index == 0 else nn.Identity())
            input_size = width


    def forward(self, x):
        for index, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = norm(layer(x))
            if index < len(self.layers) - 1:
                x = getattr(torch.nn.functional, self.activation)(x)
        return x
