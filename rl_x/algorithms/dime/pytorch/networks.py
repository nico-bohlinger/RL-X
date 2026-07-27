import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ScorePolicy(nn.Module):
    def __init__(self, action_dimension, timestep_embed_dim, hidden_dims, output_scale, initial_timestep, initial_friction, policy_observation_indices, device):
        super().__init__()
        self.register_buffer("policy_observation_indices", torch.tensor(policy_observation_indices, dtype=torch.long, device=device))
        self.register_buffer("timestep_coefficients", torch.linspace(0.1, 100.0, timestep_embed_dim, device=device)[None])
        self.log_timestep = nn.Parameter(torch.full((1,), np.log(np.expm1(initial_timestep))))
        self.log_friction = nn.Parameter(torch.full((action_dimension,), np.log(np.expm1(initial_friction))))
        self.timestep_phase = nn.Parameter(torch.zeros((1, timestep_embed_dim)))
        self.timestep_dense_1 = nn.Linear(2 * timestep_embed_dim, timestep_embed_dim)
        self.timestep_dense_2 = nn.Linear(timestep_embed_dim, timestep_embed_dim)
        layers = []
        input_dimension = action_dimension + len(policy_observation_indices) + timestep_embed_dim
        for hidden_dimension in hidden_dims:
            layers.extend([nn.Linear(input_dimension, hidden_dimension), nn.GELU()])
            input_dimension = hidden_dimension
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(input_dimension, action_dimension)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=1.0 / np.sqrt(module.in_features))
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.output_layer.weight, output_scale)


    def forward(self, observation, action, timestep):
        observation = observation[..., self.policy_observation_indices]
        phase = self.timestep_phase.to(timestep.dtype)
        coefficients = self.timestep_coefficients.to(timestep.dtype)
        timestep_embedding = torch.cat([torch.sin(coefficients * timestep + phase), torch.cos(coefficients * timestep + phase)], dim=-1)
        timestep_embedding = self.timestep_dense_2(F.gelu(self.timestep_dense_1(timestep_embedding)))
        return torch.clamp(self.output_layer(self.network(torch.cat([action, observation, timestep_embedding], dim=-1))), -1e4, 1e4)


class DistributionalCritic(nn.Module):
    def __init__(self, action_dimension, hidden_dims, nr_atoms, momentum, warmup_steps, critic_observation_indices, device):
        super().__init__()
        self.register_buffer("critic_observation_indices", torch.tensor(critic_observation_indices, dtype=torch.long, device=device))
        input_dimension = len(critic_observation_indices) + action_dimension
        self.input_batch_renorm = BatchRenorm(input_dimension, momentum, warmup_steps)
        self.layers = nn.ModuleList()
        for hidden_dimension in hidden_dims:
            linear = nn.Linear(input_dimension, hidden_dimension)
            nn.init.normal_(linear.weight, std=1.0 / np.sqrt(linear.in_features))
            nn.init.zeros_(linear.bias)
            self.layers.append(nn.ModuleList([linear, BatchRenorm(hidden_dimension, momentum, warmup_steps)]))
            input_dimension = hidden_dimension
        self.output_layer = nn.Linear(input_dimension, nr_atoms)
        nn.init.normal_(self.output_layer.weight, std=1.0 / np.sqrt(self.output_layer.in_features))
        nn.init.zeros_(self.output_layer.bias)


    def forward(self, observation, action, train):
        x = torch.cat([observation[..., self.critic_observation_indices], action], dim=-1)
        x = self.input_batch_renorm(x, train)
        for linear, batch_renorm in self.layers:
            x = batch_renorm(F.relu(linear(x)), train)
        return torch.softmax(self.output_layer(x), dim=-1)


class VectorDistributionalCritic(nn.Module):
    def __init__(self, nr_critics, action_dimension, hidden_dims, nr_atoms, momentum, warmup_steps, critic_observation_indices, device):
        super().__init__()
        self.critics = nn.ModuleList([DistributionalCritic(action_dimension, hidden_dims, nr_atoms, momentum, warmup_steps, critic_observation_indices, device) for unused_index in range(nr_critics)])


    def forward(self, observation, action, train):
        return torch.stack([critic(observation, action, train) for critic in self.critics], dim=0)


class EntropyCoefficient(nn.Module):
    def __init__(self, initial_value):
        super().__init__()
        self.log_coefficient = nn.Parameter(torch.tensor(np.log(initial_value), dtype=torch.float32))


    def forward(self):
        return torch.exp(self.log_coefficient)
