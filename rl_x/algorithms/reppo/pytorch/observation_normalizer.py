import torch
import torch.nn as nn


class ObservationNormalizer(nn.Module):
    def __init__(self, observation_shape, enabled, device):
        super().__init__()
        self.enabled = enabled
        self.register_buffer("mean", torch.zeros(observation_shape, dtype=torch.float32, device=device))
        self.register_buffer("var", torch.ones(observation_shape, dtype=torch.float32, device=device))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32, device=device))


    @torch.no_grad()
    def update(self, observations):
        if not self.enabled:
            return
        batch_mean = observations.mean(dim=0)
        batch_var = observations.var(dim=0, unbiased=False)
        batch_count = observations.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self.mean.copy_(mean)
        self.var.copy_(m_2 / total_count)
        self.count.copy_(total_count)


    def normalize(self, observations):
        if not self.enabled:
            return observations
        return (observations - self.mean) / torch.sqrt(self.var + 1e-8)
