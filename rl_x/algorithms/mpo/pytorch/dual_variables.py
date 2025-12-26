import torch


class DualVariables(torch.nn.Module):
    def __init__(self, config, nr_actions, device):
        super().__init__()
        self.log_eta = torch.nn.Parameter(torch.tensor([config.algorithm.init_log_eta], device=device))
        self.log_alpha_mean = torch.nn.Parameter(torch.tensor([config.algorithm.init_log_alpha_mean] * nr_actions, device=device))
        self.log_alpha_stddev = torch.nn.Parameter(torch.tensor([config.algorithm.init_log_alpha_stddev] * nr_actions, device=device))
        self.log_penalty_temperature = torch.nn.Parameter(torch.tensor([config.algorithm.init_log_penalty_temperature], device=device))
