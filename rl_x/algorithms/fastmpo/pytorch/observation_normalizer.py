import torch
import torch.nn as nn


class ObservationNormalizer(nn.Module):
    def __init__(self, observation_size, enable_observation_normalization, epsilon):
        super().__init__()
        self.enable_observation_normalization = enable_observation_normalization
        self.epsilon = epsilon
        if enable_observation_normalization:
            self.register_buffer("running_mean", torch.zeros((1, observation_size)))
            self.register_buffer("running_var", torch.ones((1, observation_size)))
            self.register_buffer("running_std_dev", torch.ones((1, observation_size)))
            self.register_buffer("count", torch.tensor(0, dtype=torch.long))


    @torch.no_grad()
    def normalize(self, observations, update=False):
        if not self.enable_observation_normalization:
            return observations
        if update and self.training:
            self.update_running_stats(observations)
        return (observations - self.running_mean) / (self.running_std_dev + self.epsilon)


    @torch.no_grad()
    def update_running_stats(self, observations):
        batch_mean = torch.mean(observations, dim=0, keepdim=True)
        batch_var = torch.var(observations, dim=0, unbiased=False, keepdim=True)
        batch_count = observations.shape[0]
        new_count = self.count + batch_count
        delta = batch_mean - self.running_mean
        self.running_mean.copy_(self.running_mean + delta * batch_count / new_count)
        delta2 = batch_mean - self.running_mean
        m2 = self.running_var * self.count + batch_var * batch_count + delta2.pow(2) * self.count * batch_count / new_count
        self.running_var.copy_(m2 / new_count)
        self.running_std_dev.copy_(self.running_var.sqrt())
        self.count.copy_(new_count)
