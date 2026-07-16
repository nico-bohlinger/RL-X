import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(
            env=env,
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_bins=config.algorithm.nr_bins,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            critic_observation_indices=critic_observation_indices,
            device=device,
        ).to(device)


class Critic(nn.Module):
    def __init__(self, env, hidden_dim, nr_bins, v_min, v_max, critic_observation_indices, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        nr_observations = len(critic_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()

        self.encoder = nn.Sequential(
            nn.Linear(nr_observations + nr_actions, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.critic_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, nr_bins),
        )
        self.pred_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.RMSNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        bin_width = (v_max - v_min) / (nr_bins - 1)
        support = torch.linspace(v_min - bin_width / 2, v_max + bin_width / 2, nr_bins + 1, dtype=torch.float32, device=device)
        cdf_evals = torch.erf(support / (np.sqrt(2) * bin_width * 0.75))
        zero_distribution = (cdf_evals[1:] - cdf_evals[:-1]) / (cdf_evals[-1] - cdf_evals[0])
        self.zero_distribution = nn.Parameter(zero_distribution)


    def forward(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = torch.cat([obs, action], dim=-1)
        features = self.encoder(x)
        critic_logits = self.critic_head(features) + 40.0 * self.zero_distribution
        pred = self.pred_head(features)
        pred_features = pred[..., 1:]
        pred_reward = pred[..., :1]
        return features, critic_logits, pred_features, pred_reward
