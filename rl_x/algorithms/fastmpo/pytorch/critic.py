import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def initialize_lecun_normal(layer):
    std = np.sqrt(1.0 / layer.in_features) / 0.8796256610342398
    nn.init.trunc_normal_(layer.weight, std=std, a=-2.0 * std, b=2.0 * std)
    nn.init.zeros_(layer.bias)


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", np.arange(env.single_observation_space.shape[0]))
    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        critic = VectorCritic(config, env, device, critic_observation_indices).to(device)
        critic.forward = torch.compile(critic.forward, mode=config.algorithm.compile_mode)
        critic.forward_target = torch.compile(critic.forward_target, mode=config.algorithm.compile_mode)
        return critic


class FastSACCritic(nn.Module):
    def __init__(self, nr_observations, nr_actions, nr_atoms):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(nr_observations + nr_actions, 768), nn.LayerNorm(768, eps=1e-6), nn.SiLU(), nn.Linear(768, 384), nn.LayerNorm(384, eps=1e-6), nn.SiLU(), nn.Linear(384, 192), nn.LayerNorm(192, eps=1e-6), nn.SiLU(), nn.Linear(192, nr_atoms))
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                initialize_lecun_normal(layer)


    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=-1))


class FastTD3Critic(nn.Module):
    def __init__(self, nr_observations, nr_actions, nr_atoms):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(nr_observations + nr_actions, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, nr_atoms))
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                initialize_lecun_normal(layer)


    def forward(self, observation, action):
        return self.network(torch.cat([observation, action], dim=-1))


class MPOCritic(nn.Module):
    def __init__(self, nr_observations, nr_actions, nr_atoms):
        super().__init__()
        self.dense1 = nn.Linear(nr_observations + nr_actions, 512)
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, nr_atoms)
        for layer in [self.dense1, self.dense2, self.dense3]:
            bound = np.sqrt(3 / layer.in_features) * 0.333
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        std = np.sqrt(1e-5 / self.output.in_features) / 0.8796256610342398
        nn.init.trunc_normal_(self.output.weight, std=std, a=-2.0 * std, b=2.0 * std)
        nn.init.zeros_(self.output.bias)


    def forward(self, observation, action):
        torso = torch.cat([observation, action], dim=-1)
        torso = torch.tanh(self.layer_norm(self.dense1(torso)))
        torso = torch.nn.functional.elu(self.dense2(torso))
        torso = torch.nn.functional.elu(self.dense3(torso))
        return self.output(torso)


class VectorCritic(nn.Module):
    def __init__(self, config, env, device, critic_observation_indices):
        super().__init__()
        self.critic_observation_indices = torch.tensor(critic_observation_indices, dtype=torch.long, device=device)
        nr_observations = len(critic_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()
        critic_class = FastSACCritic if config.algorithm.critic_network_type == "fastsac" else FastTD3Critic if config.algorithm.critic_network_type == "fasttd3" else MPOCritic
        nr_critics = 2 if config.algorithm.dual_critic else 1
        self.q_networks = nn.ModuleList([critic_class(nr_observations, nr_actions, config.algorithm.nr_atoms) for _ in range(nr_critics)])
        self.target_q_networks = nn.ModuleList([critic_class(nr_observations, nr_actions, config.algorithm.nr_atoms) for _ in range(nr_critics)])
        self.target_q_networks.load_state_dict(self.q_networks.state_dict())
        for parameter in self.target_q_networks.parameters():
            parameter.requires_grad_(False)


    def forward(self, observation, action):
        observation = observation[..., self.critic_observation_indices]
        return torch.stack([network(observation, action) for network in self.q_networks])


    def forward_target(self, observation, action):
        observation = observation[..., self.critic_observation_indices]
        return torch.stack([network(observation, action) for network in self.target_q_networks])
