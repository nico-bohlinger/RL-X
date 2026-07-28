import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.crossq.pytorch.batch_renorm import BatchRenorm


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = torch.compile(Policy(config, env, device, policy_observation_indices).to(device), mode=config.algorithm.compile_mode)
        policy.forward = torch.compile(policy.forward, mode=config.algorithm.compile_mode)
        policy.get_action = torch.compile(policy.get_action, mode=config.algorithm.compile_mode)
        policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
        return policy


class Policy(nn.Module):
    def __init__(self, config, env, device, policy_observation_indices):
        super().__init__()
        self.log_std_min = config.algorithm.log_std_min
        self.log_std_max = config.algorithm.log_std_max
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        input_dim = len(policy_observation_indices)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.input_batch_renorm = BatchRenorm(input_dim, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.hidden_1 = nn.Linear(input_dim, config.algorithm.policy_nr_hidden_units)
        self.batch_renorm_1 = BatchRenorm(config.algorithm.policy_nr_hidden_units, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.hidden_2 = nn.Linear(config.algorithm.policy_nr_hidden_units, config.algorithm.policy_nr_hidden_units)
        self.batch_renorm_2 = BatchRenorm(config.algorithm.policy_nr_hidden_units, config.algorithm.batch_renorm_momentum, config.algorithm.batch_renorm_warmup_steps)
        self.mean = nn.Linear(config.algorithm.policy_nr_hidden_units, action_dim)
        self.log_std = nn.Linear(config.algorithm.policy_nr_hidden_units, action_dim)


    def forward(self, x, train):
        x = x[..., self.policy_observation_indices]
        x = self.input_batch_renorm(x, train)
        x = self.batch_renorm_1(torch.relu(self.hidden_1(x)), train)
        x = self.batch_renorm_2(torch.relu(self.hidden_2(x)), train)
        return self.mean(x), torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)


    def get_action(self, x, train=False):
        mean, log_std = self(x, train)
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        pretanh = mean + std * noise
        action = torch.tanh(pretanh)
        log_prob = -0.5 * noise ** 2 - 0.5 * np.log(2.0 * np.pi) - log_std
        log_prob -= torch.log(1.0 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        processed_action = self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
        return action, processed_action, log_prob


    def get_deterministic_action(self, x):
        mean, _ = self(x, False)
        action = torch.tanh(mean)
        return self.env_as_low + 0.5 * (action + 1.0) * (self.env_as_high - self.env_as_low)
