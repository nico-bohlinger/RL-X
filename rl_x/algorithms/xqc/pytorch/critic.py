import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.xqc.pytorch.layers import BNEmbedder, XQCBlock


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(VectorCritic(config, env, device).to(device), mode=config.algorithm.compile_mode)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        input_dim = len(critic_observation_indices) + np.prod(env.single_action_space.shape, dtype=int).item()

        self.embedder = BNEmbedder(input_dim)
        self.blocks = nn.ModuleList()
        for _ in range(config.algorithm.critic_nr_blocks):
            self.blocks.append(XQCBlock(input_dim, config.algorithm.critic_hidden_dim, config.algorithm.skip_connections))
            input_dim = config.algorithm.critic_hidden_dim
        self.value = nn.Linear(input_dim, self.nr_atoms)
        nn.init.orthogonal_(self.value.weight, np.sqrt(2))
        nn.init.zeros_(self.value.bias)


    def forward(self, obs, action, train, update_stats=True):
        obs = obs[..., self.critic_observation_indices]
        x = torch.cat([obs, action], dim=-1)
        x = self.embedder(x, train, update_stats)
        for block in self.blocks:
            x = block(x, train, update_stats)
        log_probs = torch.log_softmax(self.value(x), dim=-1)
        bin_values = torch.linspace(self.v_min, self.v_max, self.nr_atoms, dtype=torch.float32, device=x.device)
        value = (torch.exp(log_probs) * bin_values).sum(dim=-1)
        return value, log_probs


class VectorCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.critics = nn.ModuleList([Critic(config, env, device) for _ in range(config.algorithm.nr_critics)])


    def forward(self, obs, action, train, update_stats=True):
        values = []
        log_probs = []
        for critic in self.critics:
            value, log_prob = critic(obs, action, train, update_stats)
            values.append(value)
            log_probs.append(log_prob)
        return torch.stack(values), torch.stack(log_probs)
