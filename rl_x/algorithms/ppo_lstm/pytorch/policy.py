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
        policy = Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.nr_hidden_units, config.algorithm.obs_encoding_dim, config.algorithm.lstm_hidden_dim, config.algorithm.lstm_obs_combine_method, config.algorithm.share_lstm_obs_encoder, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, nr_hidden_units, obs_encoding_dim, lstm_hidden_dim, lstm_obs_combine_method, share_lstm_obs_encoder, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_obs_combine_method = lstm_obs_combine_method
        self.share_lstm_obs_encoder = share_lstm_obs_encoder
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.lstm_obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
        self.lstm_obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)
        if not self.share_lstm_obs_encoder:
            self.obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
            self.obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)

        self.lstm_ii = nn.Linear(obs_encoding_dim, lstm_hidden_dim, bias=False)
        self.lstm_if = nn.Linear(obs_encoding_dim, lstm_hidden_dim, bias=False)
        self.lstm_ig = nn.Linear(obs_encoding_dim, lstm_hidden_dim, bias=False)
        self.lstm_io = nn.Linear(obs_encoding_dim, lstm_hidden_dim, bias=False)
        self.lstm_hi = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_hf = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_hg = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        self.lstm_ho = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        for layer in [self.lstm_ii, self.lstm_if, self.lstm_ig, self.lstm_io]:
            nn.init.normal_(layer.weight, std=1 / np.sqrt(obs_encoding_dim))
        for layer in [self.lstm_hi, self.lstm_hf, self.lstm_hg, self.lstm_ho]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.lstm_ln = nn.LayerNorm(lstm_hidden_dim, eps=1e-6)

        if self.lstm_obs_combine_method == "film":
            self.lstm_film_gamma = self.layer_init(nn.Linear(lstm_hidden_dim, obs_encoding_dim))
            self.lstm_film_beta = self.layer_init(nn.Linear(lstm_hidden_dim, obs_encoding_dim))

        torso_input_dim = obs_encoding_dim + lstm_hidden_dim if self.lstm_obs_combine_method == "concat" else obs_encoding_dim
        self.torso_dense1 = self.layer_init(nn.Linear(torso_input_dim, nr_hidden_units))
        self.torso_dense2 = self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))
        self.mean_head = self.layer_init(nn.Linear(nr_hidden_units, action_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def initialize_carry(self, nr_envs):
        return (torch.zeros((nr_envs, self.lstm_hidden_dim), dtype=torch.float32, device=self.policy_logstd.device), torch.zeros((nr_envs, self.lstm_hidden_dim), dtype=torch.float32, device=self.policy_logstd.device))


    def apply_one_step(self, obs, carry):
        obs = obs[..., self.policy_observation_indices]
        cell, hidden = carry
        lstm_obs_latent = torch.tanh(self.lstm_obs_encoder_ln(self.lstm_obs_encoder_dense(obs)))
        input_gate = torch.sigmoid(self.lstm_ii(lstm_obs_latent) + self.lstm_hi(hidden))
        forget_gate = torch.sigmoid(self.lstm_if(lstm_obs_latent) + self.lstm_hf(hidden))
        candidate = torch.tanh(self.lstm_ig(lstm_obs_latent) + self.lstm_hg(hidden))
        output_gate = torch.sigmoid(self.lstm_io(lstm_obs_latent) + self.lstm_ho(hidden))
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)

        if self.share_lstm_obs_encoder:
            obs_latent = lstm_obs_latent
        else:
            obs_latent = torch.tanh(self.obs_encoder_ln(self.obs_encoder_dense(obs)))
        lstm_latent = torch.tanh(self.lstm_ln(hidden))
        if self.lstm_obs_combine_method == "concat":
            torso = torch.cat([obs_latent, lstm_latent], dim=-1)
        else:
            torso = obs_latent * self.lstm_film_gamma(lstm_latent) + self.lstm_film_beta(lstm_latent)
        torso = torch.tanh(self.torso_dense1(torso))
        torso = torch.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.policy_logstd.expand_as(mean), (cell, hidden)


    def forward_sequence(self, obs, dones, carry):
        means = []
        for step in range(obs.shape[0]):
            if step > 0:
                carry = tuple(value * (1 - dones[step - 1].unsqueeze(-1)) for value in carry)
            mean, logstd, carry = self.apply_one_step(obs[step], carry)
            means.append(mean)
        means = torch.stack(means)
        return means, self.policy_logstd.expand_as(means)


    def get_action_logprob(self, obs, carry):
        mean, logstd, carry = self.apply_one_step(obs, carry)
        distribution = Normal(mean, logstd.exp())
        action = distribution.sample()
        if self.action_clipping_and_rescaling:
            processed_action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        else:
            processed_action = action
        return action, processed_action, distribution.log_prob(action).sum(-1), carry


    def get_logprob_entropy(self, obs, dones, carry, action):
        mean, logstd = self.forward_sequence(obs, dones, carry)
        distribution = Normal(mean, logstd.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


    def get_deterministic_action(self, obs, carry):
        action, logstd, carry = self.apply_one_step(obs, carry)
        if self.action_clipping_and_rescaling:
            action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        return action, carry
