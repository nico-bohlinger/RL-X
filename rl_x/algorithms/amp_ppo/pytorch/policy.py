import math
import torch
from torch import nn
from rl_x.algorithms.amp_ppo.pytorch.network import Network


class Policy(nn.Module):
    def __init__(self, config, env):
        super().__init__()
        indices = getattr(env, "policy_observation_indices", range(env.single_observation_space.shape[0]))
        self.register_buffer("observation_indices", torch.as_tensor(list(indices), dtype=torch.long))
        self.network = Network(len(indices), [512, 256, 128, env.single_action_space.shape[0]], 0.01, layer_norm=True)
        self.policy_logstd = nn.Parameter(torch.full((1, env.single_action_space.shape[0]), math.log(config.algorithm.std_dev)))


    def forward(self, x):
        return self.network(x[..., self.observation_indices]), self.policy_logstd
