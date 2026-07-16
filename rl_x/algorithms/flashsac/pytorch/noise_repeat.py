import torch


def build_zeta_cdf(mu, max_n, device):
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / pmf.sum()
    return torch.cumsum(pmf, dim=0)


class NoiseRepeat:
    def __init__(self, nr_envs, action_dim, mu, max_n, device):
        self.zeta_cdf = build_zeta_cdf(mu, max_n, device)
        self.device = device
        self.noise = torch.randn((nr_envs, action_dim), device=device)
        self.count = torch.zeros((), dtype=torch.int32, device=device)
        self.n = torch.ones((), dtype=torch.int32, device=device)


    @torch.no_grad()
    def step(self):
        new_noise = torch.randn_like(self.noise)
        u = torch.rand((), device=self.device)
        new_n = ((u < self.zeta_cdf).int().argmax() + 1).to(torch.int32)
        reinit = (self.count == 0) | (self.count >= self.n)
        self.noise = torch.where(reinit, new_noise, self.noise)
        self.n = torch.where(reinit, new_n, self.n)
        self.count = torch.where(reinit, torch.zeros_like(self.count), self.count) + 1
        return self.noise
