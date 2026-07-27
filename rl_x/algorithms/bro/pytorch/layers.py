import numpy as np
import torch
import torch.nn as nn


class BroNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, nr_blocks, output_nodes=0):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.hidden_1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(nr_blocks)])
        self.hidden_1_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(nr_blocks)])
        self.hidden_2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(nr_blocks)])
        self.hidden_2_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(nr_blocks)])
        self.output = nn.Linear(hidden_dim, output_nodes) if output_nodes > 0 else None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, np.sqrt(2.0))
                nn.init.zeros_(module.bias)


    def forward(self, x):
        x = torch.relu(self.input_norm(self.input(x)))
        for hidden_1, hidden_1_norm, hidden_2, hidden_2_norm in zip(self.hidden_1, self.hidden_1_norm, self.hidden_2, self.hidden_2_norm):
            residual = torch.relu(hidden_1_norm(hidden_1(x)))
            x = hidden_2_norm(hidden_2(residual)) + x
        return self.output(x) if self.output is not None else x
