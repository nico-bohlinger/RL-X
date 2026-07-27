import numpy as np
import torch
import torch.nn as nn


def get_entropy_coefficient(config, env, device):
    compile_mode = config.algorithm.compile_mode
    entropy_coefficient = torch.compile(EntropyCoefficient(config, env, device).to(device), mode=compile_mode)
    entropy_coefficient.forward = torch.compile(entropy_coefficient.forward, mode=compile_mode)
    entropy_coefficient.loss = torch.compile(entropy_coefficient.loss, mode=compile_mode)
    return entropy_coefficient


class EntropyCoefficient(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.target_entropy = config.algorithm.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -torch.prod(torch.tensor(np.prod(env.single_action_space.shape), dtype=torch.float32).to(device)).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
    
    
    def forward(self):
        return self.log_alpha.exp()
    

    def loss(self, entropy):
        return self.log_alpha.exp() * (entropy - self.target_entropy)
