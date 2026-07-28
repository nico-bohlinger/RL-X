import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env, device):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", np.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        policy = Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.obs_encoding_dim, config.algorithm.window_length, config.algorithm.window_hidden_dim, config.algorithm.window_obs_combine_method, config.algorithm.share_window_obs_encoder, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, obs_encoding_dim, window_length, window_hidden_dim, window_obs_combine_method, share_window_obs_encoder, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.window_length = window_length
        self.window_obs_combine_method = window_obs_combine_method
        self.share_window_obs_encoder = share_window_obs_encoder
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.window_obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
        self.window_obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)
        if not self.share_window_obs_encoder:
            self.obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
            self.obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)
        self.window_agg_dense = self.layer_init(nn.Linear(window_length * obs_encoding_dim, window_hidden_dim))
        self.window_agg_ln = nn.LayerNorm(window_hidden_dim, eps=1e-6)
        if self.window_obs_combine_method == "film":
            self.window_film_gamma = self.layer_init(nn.Linear(window_hidden_dim, obs_encoding_dim))
            self.window_film_beta = self.layer_init(nn.Linear(window_hidden_dim, obs_encoding_dim))
        torso_input_dim = obs_encoding_dim + window_hidden_dim if self.window_obs_combine_method == "concat" else obs_encoding_dim
        self.torso_dense1 = self.layer_init(nn.Linear(torso_input_dim, 512))
        self.torso_ln1 = nn.LayerNorm(512, eps=1e-6)
        self.torso_dense2 = self.layer_init(nn.Linear(512, 256))
        self.torso_dense3 = self.layer_init(nn.Linear(256, 128))
        self.mean_head = self.layer_init(nn.Linear(128, action_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def initialize_window(self, nr_envs, obs_dim):
        return torch.zeros((nr_envs, self.window_length, obs_dim), dtype=torch.float32, device=self.policy_logstd.device)


    def apply_one_step(self, obs, window):
        indexed_obs = obs[..., self.policy_observation_indices]
        window_latent = torch.nn.functional.elu(self.window_obs_encoder_ln(self.window_obs_encoder_dense(window[..., self.policy_observation_indices])))
        window_latent = torch.nn.functional.elu(self.window_agg_ln(self.window_agg_dense(window_latent.flatten(start_dim=-2))))
        if self.share_window_obs_encoder:
            obs_latent = torch.nn.functional.elu(self.window_obs_encoder_ln(self.window_obs_encoder_dense(indexed_obs)))
        else:
            obs_latent = torch.nn.functional.elu(self.obs_encoder_ln(self.obs_encoder_dense(indexed_obs)))
        if self.window_obs_combine_method == "concat":
            torso = torch.cat([obs_latent, window_latent], dim=-1)
        else:
            torso = obs_latent * self.window_film_gamma(window_latent) + self.window_film_beta(window_latent)
        torso = torch.nn.functional.elu(self.torso_ln1(self.torso_dense1(torso)))
        torso = torch.nn.functional.elu(self.torso_dense2(torso))
        torso = torch.nn.functional.elu(self.torso_dense3(torso))
        mean = self.mean_head(torso)
        next_window = torch.cat([window[..., 1:, :], obs.unsqueeze(-2)], dim=-2)
        return mean, self.policy_logstd.expand_as(mean), next_window


    def forward_sequence(self, obs, dones, window):
        means = []
        for step in range(obs.shape[0]):
            if step > 0:
                window = torch.where(dones[step - 1, :, None, None] > 0, torch.zeros_like(window), window)
            mean, logstd, window = self.apply_one_step(obs[step], window)
            means.append(mean)
        means = torch.stack(means)
        return means, self.policy_logstd.expand_as(means)


    def get_action_logprob(self, obs, window):
        mean, logstd, window = self.apply_one_step(obs, window)
        distribution = Normal(mean, logstd.exp())
        action = distribution.sample()
        if self.action_clipping_and_rescaling:
            processed_action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        else:
            processed_action = action
        return action, processed_action, distribution.log_prob(action).sum(-1), window


    def get_logprob_entropy(self, obs, dones, window, action):
        mean, logstd = self.forward_sequence(obs, dones, window)
        distribution = Normal(mean, logstd.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


    def get_deterministic_action(self, obs, window):
        action, logstd, window = self.apply_one_step(obs, window)
        if self.action_clipping_and_rescaling:
            action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        return action, window
