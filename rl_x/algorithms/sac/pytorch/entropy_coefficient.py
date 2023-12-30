import numpy as np
import torch
import torch.nn as nn


class EntropyCoefficient(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.target_entropy = config.algorithm.target_entropy
        if self.target_entropy == "auto":
            self.target_entropy = -torch.prod(torch.tensor(np.prod(env.single_action_space.shape), dtype=torch.float32).to(device)).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
    
    
    @torch.jit.export
    def forward(self):
        return self.log_alpha.exp()


    @torch.jit.export
    def loss(self, entropy):
        alpha = self.log_alpha.exp()
        entropy_loss = alpha * (entropy - self.target_entropy)

        return entropy_loss
