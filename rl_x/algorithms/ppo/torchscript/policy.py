import numpy as np
import torch
import torch.nn as nn

from rl_x.algorithms.ppo.torchscript.normal_distribution import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
        env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        return Policy(env, config.algorithm.std_dev, config.algorithm.nr_hidden_units, env_as_low, env_as_high,
                     config.algorithm.clip_range, config.algorithm.entropy_coef)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Policy(nn.Module):
    def __init__(self, env, std_dev, nr_hidden_units, env_as_low, env_as_high, clip_range: float, entropy_coef: float):
        super().__init__()
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.policy_as_low = -1
        self.policy_as_high = 1
        self.env_as_low = env_as_low
        self.env_as_high = env_as_high
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        self.policy_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, np.prod(single_as_shape).item()), std=0.01),
        )
        self.policy_logstd = nn.Parameter(torch.full((1, np.prod(single_as_shape).item()), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2).item(), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    @torch.jit.export
    def get_action_logprob(self, x):
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return action, clipped_and_scaled_action, probs.log_prob(action).sum(1)
    

    @torch.jit.export
    def get_logprob_entropy(self, x, action):
        action_mean = self.policy_mean(x)
        action_logstd = self.policy_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    @torch.jit.export
    def get_deterministic_action(self, x):
        with torch.no_grad():
            action = self.policy_mean(x)
        clipped_action = torch.clip(action, self.policy_as_low, self.policy_as_high)
        clipped_and_scaled_action = self.env_as_low + (0.5 * (clipped_action + 1.0) * (self.env_as_high - self.env_as_low))
        return clipped_and_scaled_action


    @torch.jit.export
    def loss(self, states, actions, log_probs, advantages):
        new_log_prob, entropy = self.get_logprob_entropy(states, actions)
        logratio = new_log_prob - log_probs
        ratio = logratio.exp()

        with torch.no_grad():
            log_ratio = new_log_prob - log_probs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())

        minibatch_advantages = advantages
        minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

        pg_loss1 = -minibatch_advantages * ratio
        pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.entropy_coef * entropy_loss

        return loss, pg_loss, entropy_loss, approx_kl_div, clip_fraction
