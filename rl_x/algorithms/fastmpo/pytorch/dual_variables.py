import torch
import torch.nn as nn


class DualVariables(nn.Module):
    def __init__(self, nr_actions, init_log_eta, init_log_alpha_mean, init_log_alpha_stddev, init_log_penalty_temperature):
        super().__init__()
        self.log_eta = nn.Parameter(torch.full((1,), init_log_eta))
        self.log_alpha_mean = nn.Parameter(torch.full((nr_actions,), init_log_alpha_mean))
        self.log_alpha_stddev = nn.Parameter(torch.full((nr_actions,), init_log_alpha_stddev))
        self.log_penalty_temperature = nn.Parameter(torch.full((1,), init_log_penalty_temperature))


    def forward(self):
        return self.log_eta, self.log_alpha_mean, self.log_alpha_stddev, self.log_penalty_temperature
