import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorDistributionalCritic(config, env, device, critic_observation_indices).to(device)


class BatchRenorm(nn.Module):
    def __init__(self, dimension, momentum, warmup_steps):
        super().__init__()
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.epsilon = 0.001
        self.scale = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))
        self.register_buffer("running_mean", torch.zeros(dimension))
        self.register_buffer("running_variance", torch.ones(dimension))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))


    def forward(self, x, train):
        if train:
            reduction_axes = tuple(range(x.ndim - 1))
            mean = torch.mean(x, dim=reduction_axes)
            variance = torch.var(x, dim=reduction_axes, correction=0)
            standard_deviation = torch.sqrt(variance + self.epsilon)
            running_standard_deviation = torch.sqrt(self.running_variance + self.epsilon)
            r = torch.clamp(standard_deviation / running_standard_deviation, 1.0 / 3.0, 3.0).detach()
            d = torch.clamp((mean - self.running_mean) / running_standard_deviation, -5.0, 5.0).detach()
            normalized = (x - mean) / standard_deviation
            normalized = torch.where(self.steps >= self.warmup_steps, normalized * r + d, normalized)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
                self.running_variance.mul_(self.momentum).add_(variance, alpha=1.0 - self.momentum)
                self.steps.add_(1)
        else:
            normalized = (x - self.running_mean) / torch.sqrt(self.running_variance + self.epsilon)
        return normalized * self.scale.to(x.dtype) + self.bias.to(x.dtype)


class DistributionalCritic(nn.Module):
    def __init__(self, config, env, device, critic_observation_indices):
        super().__init__()
        action_dimension = np.prod(env.single_action_space.shape, dtype=int).item()
        self.register_buffer("critic_observation_indices", torch.tensor(critic_observation_indices, dtype=torch.long, device=device))
        input_dimension = len(critic_observation_indices) + action_dimension
        self.input_batch_renorm = BatchRenorm(input_dimension, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.layers = nn.ModuleList()
        for hidden_dimension in config.algorithm.critic_hidden_dims:
            linear = nn.Linear(input_dimension, hidden_dimension)
            nn.init.normal_(linear.weight, std=1.0 / np.sqrt(linear.in_features))
            nn.init.zeros_(linear.bias)
            self.layers.append(nn.ModuleList([linear, BatchRenorm(hidden_dimension, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)]))
            input_dimension = hidden_dimension
        self.output_layer = nn.Linear(input_dimension, config.algorithm.nr_atoms)
        nn.init.normal_(self.output_layer.weight, std=1.0 / np.sqrt(self.output_layer.in_features))
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, observation, action, train):
        x = torch.cat([observation[..., self.critic_observation_indices], action], dim=-1)
        x = self.input_batch_renorm(x, train)
        for linear, batch_renorm in self.layers:
            x = batch_renorm(F.relu(linear(x)), train)
        return torch.softmax(self.output_layer(x), dim=-1)


class VectorDistributionalCritic(nn.Module):
    def __init__(self, config, env, device, critic_observation_indices):
        super().__init__()
        self.critics = nn.ModuleList([DistributionalCritic(config, env, device, critic_observation_indices) for unused_index in range(config.algorithm.nr_critics)])


    def forward(self, observation, action, train):
        return torch.stack([critic(observation, action, train) for critic in self.critics], dim=0)
