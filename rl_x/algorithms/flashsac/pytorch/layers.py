import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight, gain=1.0)


    def forward(self, x):
        return F.linear(x, self.weight)


    @torch.no_grad()
    def normalize(self):
        self.weight.data.copy_(F.normalize(self.weight.data, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.01, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.momentum = momentum
        self.eps = eps


    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=self.training, momentum=self.momentum, eps=self.eps)


    @torch.no_grad()
    def normalize(self):
        scale, bias = self.weight.data, self.bias.data
        d = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        factor = math.sqrt(d) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * factor)
        self.bias.data.copy_(bias * factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eps = eps


    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, eps=self.eps)


    @torch.no_grad()
    def normalize(self):
        scale = self.weight.data
        d = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        factor = math.sqrt(d) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * factor)


class FlashSACEmbedder(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.norm = UnitBatchNorm(in_features)
        self.linear = UnitLinear(in_features, hidden_dim)


    def forward(self, x):
        return self.linear(self.norm(x))


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim, expansion=4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm2 = UnitBatchNorm(hidden_dim)


    def forward(self, x):
        residual = x
        h = F.relu(self.norm1(self.w1(x)))
        h = F.relu(self.norm2(self.w2(h)))
        return h + residual


class EnsembleCategoricalValue(nn.Module):
    def __init__(self, nr_critics, hidden_dim, nr_atoms, v_min, v_max):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nr_critics, nr_atoms, hidden_dim))
        for i in range(nr_critics):
            nn.init.orthogonal_(self.weight.data[i], gain=1.0)
        self.bias = nn.Parameter(torch.zeros(nr_critics, nr_atoms))
        self.register_buffer("bin_values", torch.linspace(v_min, v_max, nr_atoms).view(1, 1, -1))


    def forward(self, x):
        logits = torch.einsum("nbh,noh->nbo", x, self.weight) + self.bias.unsqueeze(1)
        log_probs = F.log_softmax(logits, dim=-1)
        value = torch.sum(torch.exp(log_probs) * self.bin_values, dim=-1)
        return value, log_probs


    @torch.no_grad()
    def normalize(self):
        self.weight.data.copy_(F.normalize(self.weight.data, dim=-1, eps=1e-8))


def normalize_module(module):
    for m in module.modules():
        if isinstance(m, (UnitLinear, UnitBatchNorm, UnitRMSNorm, EnsembleCategoricalValue)):
            m.normalize()
