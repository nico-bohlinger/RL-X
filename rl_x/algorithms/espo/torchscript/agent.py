import numpy as np
import torch
import torch.nn as nn

from rl_x.algorithms.ppo.torchscript.normal_distribution import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_agent(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Agent(env, config.algorithm.std_dev, config.algorithm.nr_hidden_units, config.algorithm.ent_coef, config.algorithm.vf_coef)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")


class Agent(nn.Module):
    def __init__(self, env, std_dev, nr_hidden_units, ent_coef: float, vf_coef: float):
        super().__init__()
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        if nr_hidden_units < 1:
            raise ValueError("nr_hidden_units must be >= 1")

        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, single_as_shape.item()), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.full((1, single_as_shape.item()), np.log(std_dev).item()))

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.prod(single_os_shape).item(), nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(nr_hidden_units, 1), std=1.0),
        )


    def layer_init(self, layer, std=np.sqrt(2).item(), bias_const=(0.0)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    

    @torch.jit.export
    def get_action_logprob(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        return action, probs.log_prob(action).sum(1)
    

    @torch.jit.export
    def get_logprob_entropy(self, x, action):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)  # (nr_envs, as_shape)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.log_prob(action).sum(1), probs.entropy().sum(1)
    

    @torch.jit.export
    def get_value(self, x):
        return self.critic(x)
    

    @torch.jit.export
    def loss(self, states, actions, log_probs, returns, advantages):
        new_log_prob, entropy = self.get_logprob_entropy(states, actions)
        new_value = self.get_value(states).reshape(-1)
        logratio = new_log_prob - log_probs
        ratio = logratio.exp()

        with torch.no_grad():
            log_ratio = new_log_prob - log_probs
            approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

        minibatch_advantages = advantages
        minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

        pg_loss = (-minibatch_advantages * ratio).mean()
        v_loss = (0.5 * (new_value - returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

        return ratio, loss, pg_loss, v_loss, entropy_loss, approx_kl_div
