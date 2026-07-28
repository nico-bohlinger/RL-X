import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.crossq.pytorch.batch_renorm import BatchRenorm


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return torch.compile(VectorCritic(config, env, device).to(device), mode=config.algorithm.compile_mode)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        input_dim = len(critic_observation_indices) + np.prod(env.single_action_space.shape, dtype=int).item()

        self.input_batch_renorm = BatchRenorm(input_dim, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.hidden_1 = nn.Linear(input_dim, config.algorithm.critic_nr_hidden_units)
        self.batch_renorm_1 = BatchRenorm(config.algorithm.critic_nr_hidden_units, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.hidden_2 = nn.Linear(config.algorithm.critic_nr_hidden_units, config.algorithm.critic_nr_hidden_units)
        self.batch_renorm_2 = BatchRenorm(config.algorithm.critic_nr_hidden_units, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.value = nn.Linear(config.algorithm.critic_nr_hidden_units, 1)


    def forward(self, obs, action, train):
        obs = obs[..., self.critic_observation_indices]
        x = torch.cat([obs, action], dim=-1)
        x = self.input_batch_renorm(x, train)
        x = self.batch_renorm_1(torch.relu(self.hidden_1(x)), train)
        x = self.batch_renorm_2(torch.relu(self.hidden_2(x)), train)
        return self.value(x)


class VectorCritic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.critics = nn.ModuleList([Critic(config, env, device) for _ in range(config.algorithm.ensemble_size)])


    def forward(self, obs, action, train):
        return torch.stack([critic(obs, action, train) for critic in self.critics])
