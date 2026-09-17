import math
import torch
from torch import nn


class EnergyFunction(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        cfg = config.algorithm
        self.ncsnv1 = cfg.ncsnv1
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        input_size = env.single_observation_space.shape[0] + (env.single_observation_space.shape[0] if cfg.state_based else env.single_action_space.shape[0])
        width = input_size
        for widths, layers, norms in [(cfg.nr_hidden_units_encoder_ncsn, self.encoder, self.encoder_norms), (cfg.nr_hidden_units_decoder_ncsn, self.decoder, self.decoder_norms)]:
            for size in widths:
                layers.append(nn.Linear(width, size))
                norms.append(nn.LayerNorm(size, eps=1e-6))
                width = size
        self.output = nn.Linear(width, 1)
        self.half_dim = cfg.nr_hidden_units_encoder_ncsn[-1] // 2
        if not self.ncsnv1:
            self.residual = nn.Linear(input_size, cfg.nr_hidden_units_encoder_ncsn[-1])
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 1.0 if layer is self.output else math.sqrt(2))
                nn.init.zeros_(layer.bias)


    def forward(self, x, condition):
        initial = x
        for layer, norm in zip(self.encoder, self.encoder_norms):
            x = norm(layer(x))
            x = torch.nn.functional.elu(x) if self.ncsnv1 else torch.nn.functional.gelu(x, approximate="tanh")
        condition = torch.as_tensor(condition, device=x.device, dtype=x.dtype)
        if self.ncsnv1:
            frequency = torch.exp(-math.log(10000) / (self.half_dim - 1) * torch.arange(self.half_dim, device=x.device, dtype=x.dtype))
            embedding = 100 * condition[..., None] * frequency
            x = x + torch.cat([embedding.sin(), embedding.cos()], dim=-1)
        else:
            x = x + self.residual(initial)
        for layer, norm in zip(self.decoder, self.decoder_norms):
            x = norm(layer(x))
            x = torch.nn.functional.elu(x) if self.ncsnv1 else torch.nn.functional.gelu(x, approximate="tanh")
        x = self.output(x).squeeze(-1)
        return torch.nn.functional.elu(x) if self.ncsnv1 else torch.nn.functional.gelu(x, approximate="tanh") / condition
