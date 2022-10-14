import numpy as np
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, os_shape):
        super().__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(os_shape, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 1), std=1.0),
        )
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(x)
