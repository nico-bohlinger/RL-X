import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BNEmbedder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim, eps=1e-3, momentum=0.01)


    def forward(self, x, train, update_stats):
        if train and not update_stats:
            return F.batch_norm(x, None, None, self.batch_norm.weight, self.batch_norm.bias, training=True, momentum=0.01, eps=1e-3)
        if train:
            return self.batch_norm(x)
        return F.batch_norm(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias, training=False, momentum=0.01, eps=1e-3)


class XQCBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, skip_connections):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.batch_norm = nn.BatchNorm1d(hidden_dim, eps=1e-3, momentum=0.01)
        self.skip_connections = skip_connections
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))


    def forward(self, x, train, update_stats):
        residual = x
        x = self.linear(x)
        if train and not update_stats:
            x = F.batch_norm(x, None, None, self.batch_norm.weight, self.batch_norm.bias, training=True, momentum=0.01, eps=1e-3)
        elif train:
            x = self.batch_norm(x)
        else:
            x = F.batch_norm(x, self.batch_norm.running_mean, self.batch_norm.running_var, self.batch_norm.weight, self.batch_norm.bias, training=False, momentum=0.01, eps=1e-3)
        x = F.relu(x)
        if self.skip_connections and residual.shape == x.shape:
            x = x + residual
        return x


def weight_norm(module, normalize_last_layer):
    with torch.no_grad():
        for name, child in module.named_modules():
            if isinstance(child, XQCBlock):
                weights = torch.cat([child.linear.weight, torch.zeros((child.linear.weight.shape[0], 1), device=child.linear.weight.device)], dim=1)
                child.linear.weight.div_(torch.linalg.vector_norm(weights, dim=1, keepdim=True))
            elif normalize_last_layer and name.split(".")[-1] in ["value", "mean", "log_std"]:
                child.weight.div_(torch.linalg.vector_norm(child.weight, dim=1, keepdim=True))
