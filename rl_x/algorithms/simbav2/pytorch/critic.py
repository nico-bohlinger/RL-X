import math
import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.simbav2.pytorch.layers import HyperDense, HyperEmbedder, HyperLERPBlock, Scaler


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(VectorCritic(config, env, device).to(device), mode=config.algorithm.compile_mode)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        self.nr_bins = config.algorithm.nr_bins
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        hidden_dim = config.algorithm.critic_hidden_dim
        scaler_init = math.sqrt(2.0 / hidden_dim)
        alpha_init = 1.0 / (config.algorithm.critic_nr_blocks + 1)
        input_dim = len(critic_observation_indices) + np.prod(env.single_action_space.shape, dtype=int).item()

        self.embedder = HyperEmbedder(input_dim, hidden_dim, scaler_init, scaler_init, config.algorithm.c_shift)
        self.blocks = nn.ModuleList([HyperLERPBlock(hidden_dim, scaler_init, scaler_init, alpha_init, 1.0 / math.sqrt(hidden_dim)) for _ in range(config.algorithm.critic_nr_blocks)])
        self.value_hidden = HyperDense(hidden_dim, hidden_dim)
        self.value_scaler = Scaler(hidden_dim, 1.0, 1.0)
        self.value = HyperDense(hidden_dim, self.nr_bins)
        self.value_bias = nn.Parameter(torch.zeros(self.nr_bins))


    def forward(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = self.embedder(torch.cat([obs, action], dim=-1))
        for block in self.blocks:
            x = block(x)
        value = self.value(self.value_scaler(self.value_hidden(x)))
        log_probs = torch.log_softmax(value + self.value_bias.to(value.dtype), dim=-1)
        bin_values = torch.linspace(self.v_min, self.v_max, self.nr_bins, dtype=torch.float32, device=x.device)
        return (torch.exp(log_probs) * bin_values).sum(dim=-1), log_probs


class VectorCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        nr_critics = 2 if config.algorithm.use_cdq else 1
        self.critics = nn.ModuleList([Critic(config, env, device) for _ in range(nr_critics)])


    def forward(self, obs, action):
        values = []
        log_probs = []
        for critic in self.critics:
            value, log_prob = critic(obs, action)
            values.append(value)
            log_probs.append(log_prob)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        return (values, log_probs) if len(self.critics) == 2 else (values[0], log_probs[0])
