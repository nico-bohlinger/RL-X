import torch
import torch.nn as nn


def get_observation_normalizer(config, observation_size, device):
    observation_normalizer = torch.compile(ObservationNormalizer(observation_size, device, config.algorithm.enable_observation_normalization).to(device), mode=config.algorithm.compile_mode)
    observation_normalizer.normalize = torch.compile(observation_normalizer.normalize, mode=config.algorithm.compile_mode)
    return observation_normalizer


class ObservationNormalizer(nn.Module):
    def __init__(self, observation_size, device, enable_observation_normalization, epsilon=1e-8):
        super().__init__()
        self.observation_size = observation_size
        self.device = device
        self.enable_observation_normalization = enable_observation_normalization
        self.epsilon = epsilon

        if self.enable_observation_normalization:
            self.register_buffer("running_mean", torch.zeros(observation_size, device=device).unsqueeze(0))
            self.register_buffer("running_var", torch.ones(observation_size, device=device).unsqueeze(0))
            self.register_buffer("running_std_dev", torch.ones(observation_size, device=device).unsqueeze(0))
            self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))


    @torch.no_grad()
    def normalize(self, observations, update=True):
        if self.enable_observation_normalization:
            if update and self.training:
                self._update_running_stats(observations)
            return (observations - self.running_mean) / (self.running_std_dev + self.epsilon)
        else:
            return observations
    

    @torch.jit.unused
    def _update_running_stats(self, observations):
        batch_mean = torch.mean(observations, dim=0, keepdim=True)
        batch_var = torch.var(observations, dim=0, unbiased=False, keepdim=True)
        batch_count = observations.shape[0]

        new_count = self.count + batch_count

        delta = batch_mean - self.running_mean
        self.running_mean.copy_(self.running_mean + delta * batch_count / new_count)

        delta2 = batch_mean - self.running_mean
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta2.pow(2) * self.count * batch_count / new_count
        self.running_var.copy_(M2 / new_count)
        self.running_std_dev.copy_(self.running_var.sqrt())
        self.count.copy_(new_count)
