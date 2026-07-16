import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.flashsac.pytorch.layers import EnsembleCategoricalValue, FlashSACBlock, FlashSACEmbedder, UnitRMSNorm


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic = DoubleCritic(
            env=env,
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_blocks=config.algorithm.critic_nr_blocks,
            nr_atoms=config.algorithm.nr_atoms,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            critic_observation_indices=critic_observation_indices,
            device=device,
        ).to(device)
        if config.algorithm.use_compile:
            critic.forward = torch.compile(critic.forward, mode=config.algorithm.compile_mode)
        return critic


class SingleCritic(nn.Module):
    def __init__(self, in_features, hidden_dim, nr_blocks):
        super().__init__()
        self.embedder = FlashSACEmbedder(in_features, hidden_dim)
        self.blocks = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(nr_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)


    def forward(self, x):
        x = self.embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_norm(x)
        return x


class DoubleCritic(nn.Module):
    def __init__(self, env, hidden_dim, nr_blocks, nr_atoms, v_min, v_max, critic_observation_indices, device):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        nr_observations = len(critic_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()
        in_features = nr_observations + nr_actions
        self.critics = nn.ModuleList([SingleCritic(in_features, hidden_dim, nr_blocks) for _ in range(2)])
        self.head = EnsembleCategoricalValue(2, hidden_dim, nr_atoms, v_min, v_max)


    def forward(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = torch.cat([obs, action], dim=-1)
        features = torch.stack([self.critics[i](x) for i in range(2)], dim=0)
        value, log_probs = self.head(features)
        return value, log_probs
