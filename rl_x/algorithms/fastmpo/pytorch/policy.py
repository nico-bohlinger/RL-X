import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def initialize_lecun_normal(layer):
    std = np.sqrt(1.0 / layer.in_features) / 0.8796256610342398
    nn.init.trunc_normal_(layer.weight, std=std, a=-2.0 * std, b=2.0 * std)
    nn.init.zeros_(layer.bias)


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        if config.algorithm.policy_network_type == "fastsac":
            policy = FastSACPolicy(config, env, device, policy_observation_indices).to(device)
        elif config.algorithm.policy_network_type == "fasttd3":
            policy = FastTD3Policy(config, env, device, policy_observation_indices).to(device)
        elif config.algorithm.policy_network_type == "mpo":
            policy = MPOPolicy(config, env, device, policy_observation_indices).to(device)

        policy.forward = torch.compile(policy.forward, mode=config.algorithm.compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=config.algorithm.compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.action_clipping = config.algorithm.action_clipping
        self.action_rescaling = config.algorithm.action_rescaling
        self.policy_init_scale = config.algorithm.policy_init_scale
        self.policy_min_scale = config.algorithm.policy_min_scale
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        env_as_scale = torch.as_tensor(env.single_action_space.scale, dtype=torch.float32, device=device)
        env_as_center = torch.as_tensor(env.single_action_space.center, dtype=torch.float32, device=device)
        self.register_buffer("env_as_low", torch.as_tensor(env.single_action_space.low, dtype=torch.float32, device=device))
        self.register_buffer("env_as_high", torch.as_tensor(env.single_action_space.high, dtype=torch.float32, device=device))
        self.register_buffer("action_scale", torch.maximum(torch.abs(self.env_as_low - env_as_center), torch.abs(self.env_as_high - env_as_center)) / env_as_scale)


    def get_processed_action(self, action):
        if self.action_clipping:
            action = torch.clamp(action, -1, 1)
        if self.action_rescaling == "normal":
            action = self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        elif self.action_rescaling == "fastsac":
            action = action * self.action_scale
        return action


    def get_action(self, observation):
        mean, stddev = self.forward(observation)
        action = mean + stddev * torch.randn_like(mean)
        return action, self.get_processed_action(action)


    def get_deterministic_action(self, observation):
        mean, stddev = self.forward(observation)
        return self.get_processed_action(mean)


class FastSACPolicy(Policy):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__(config, env, device, policy_observation_indices)
        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()
        self.torso = nn.Sequential(nn.Linear(nr_observations, 512), nn.LayerNorm(512, eps=1e-6), nn.SiLU(), nn.Linear(512, 256), nn.LayerNorm(256, eps=1e-6), nn.SiLU(), nn.Linear(256, 128), nn.LayerNorm(128, eps=1e-6), nn.SiLU())
        self.mean = nn.Linear(128, nr_actions)
        self.stddev = nn.Linear(128, nr_actions)
        for layer in self.torso:
            if isinstance(layer, nn.Linear):
                initialize_lecun_normal(layer)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.stddev.weight)
        nn.init.zeros_(self.stddev.bias)


    def forward(self, observation):
        torso = self.torso(observation[..., self.policy_observation_indices])
        return self.mean(torso), self.policy_min_scale + torch.nn.functional.softplus(self.stddev(torso)) * self.policy_init_scale / torch.nn.functional.softplus(torch.zeros((), device=observation.device))


class FastTD3Policy(Policy):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__(config, env, device, policy_observation_indices)
        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()
        self.torso = nn.Sequential(nn.Linear(nr_observations, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.mean = nn.Linear(128, nr_actions)
        self.stddev = nn.Linear(128, nr_actions)
        for layer in self.torso:
            if isinstance(layer, nn.Linear):
                initialize_lecun_normal(layer)
        nn.init.normal_(self.mean.weight, std=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.stddev.weight)
        nn.init.zeros_(self.stddev.bias)


    def forward(self, observation):
        torso = self.torso(observation[..., self.policy_observation_indices])
        return self.mean(torso), self.policy_min_scale + torch.nn.functional.softplus(self.stddev(torso)) * self.policy_init_scale / torch.nn.functional.softplus(torch.zeros((), device=observation.device))


class MPOPolicy(Policy):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__(config, env, device, policy_observation_indices)
        nr_observations = len(policy_observation_indices)
        nr_actions = np.prod(env.single_action_space.shape, dtype=int).item()
        self.dense1 = nn.Linear(nr_observations, 512)
        self.layer_norm = nn.LayerNorm(512, eps=1e-6)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.mean = nn.Linear(128, nr_actions)
        self.stddev = nn.Linear(128, nr_actions)
        for layer in [self.dense1, self.dense2, self.dense3]:
            bound = np.sqrt(3 / layer.in_features) * 0.333
            nn.init.uniform_(layer.weight, -bound, bound)
            nn.init.zeros_(layer.bias)
        for layer in [self.mean, self.stddev]:
            std = np.sqrt(1e-4 / layer.in_features) / 0.8796256610342398
            nn.init.trunc_normal_(layer.weight, std=std, a=-2.0 * std, b=2.0 * std)
            nn.init.zeros_(layer.bias)


    def forward(self, observation):
        torso = torch.tanh(self.layer_norm(self.dense1(observation[..., self.policy_observation_indices])))
        torso = torch.nn.functional.elu(self.dense2(torso))
        torso = torch.nn.functional.elu(self.dense3(torso))
        return self.mean(torso), self.policy_min_scale + torch.nn.functional.softplus(self.stddev(torso)) * self.policy_init_scale / torch.nn.functional.softplus(torch.zeros((), device=observation.device))
