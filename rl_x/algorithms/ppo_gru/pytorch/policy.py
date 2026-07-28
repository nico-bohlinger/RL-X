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
        policy = Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.nr_hidden_units, config.algorithm.obs_encoding_dim, config.algorithm.gru_hidden_dim, config.algorithm.gru_obs_combine_method, config.algorithm.share_gru_obs_encoder, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, nr_hidden_units, obs_encoding_dim, gru_hidden_dim, gru_obs_combine_method, share_gru_obs_encoder, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_obs_combine_method = gru_obs_combine_method
        self.share_gru_obs_encoder = share_gru_obs_encoder
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.gru_obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
        self.gru_obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)
        if not self.share_gru_obs_encoder:
            self.obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), obs_encoding_dim))
            self.obs_encoder_ln = nn.LayerNorm(obs_encoding_dim, eps=1e-6)

        self.gru_ir = nn.Linear(obs_encoding_dim, gru_hidden_dim)
        self.gru_iz = nn.Linear(obs_encoding_dim, gru_hidden_dim)
        self.gru_in = nn.Linear(obs_encoding_dim, gru_hidden_dim)
        self.gru_hr = nn.Linear(gru_hidden_dim, gru_hidden_dim, bias=False)
        self.gru_hz = nn.Linear(gru_hidden_dim, gru_hidden_dim, bias=False)
        self.gru_hn = nn.Linear(gru_hidden_dim, gru_hidden_dim)
        for layer in [self.gru_ir, self.gru_iz, self.gru_in]:
            nn.init.normal_(layer.weight, std=1 / np.sqrt(obs_encoding_dim))
            nn.init.zeros_(layer.bias)
        for layer in [self.gru_hr, self.gru_hz, self.gru_hn]:
            nn.init.orthogonal_(layer.weight)
        nn.init.zeros_(self.gru_hn.bias)
        self.gru_ln = nn.LayerNorm(gru_hidden_dim, eps=1e-6)

        if self.gru_obs_combine_method == "film":
            self.gru_film_gamma = self.layer_init(nn.Linear(gru_hidden_dim, obs_encoding_dim))
            self.gru_film_beta = self.layer_init(nn.Linear(gru_hidden_dim, obs_encoding_dim))

        torso_input_dim = obs_encoding_dim + gru_hidden_dim if self.gru_obs_combine_method == "concat" else obs_encoding_dim
        self.torso_dense1 = self.layer_init(nn.Linear(torso_input_dim, nr_hidden_units))
        self.torso_dense2 = self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))
        self.mean_head = self.layer_init(nn.Linear(nr_hidden_units, action_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def initialize_carry(self, nr_envs):
        return torch.zeros((nr_envs, self.gru_hidden_dim), dtype=torch.float32, device=self.policy_logstd.device)


    def apply_one_step(self, obs, carry):
        obs = obs[..., self.policy_observation_indices]
        gru_obs_latent = torch.tanh(self.gru_obs_encoder_ln(self.gru_obs_encoder_dense(obs)))
        reset = torch.sigmoid(self.gru_ir(gru_obs_latent) + self.gru_hr(carry))
        update = torch.sigmoid(self.gru_iz(gru_obs_latent) + self.gru_hz(carry))
        candidate = torch.tanh(self.gru_in(gru_obs_latent) + reset * self.gru_hn(carry))
        carry = (1 - update) * candidate + update * carry

        if self.share_gru_obs_encoder:
            obs_latent = gru_obs_latent
        else:
            obs_latent = torch.tanh(self.obs_encoder_ln(self.obs_encoder_dense(obs)))
        gru_latent = torch.tanh(self.gru_ln(carry))
        if self.gru_obs_combine_method == "concat":
            torso = torch.cat([obs_latent, gru_latent], dim=-1)
        else:
            torso = obs_latent * self.gru_film_gamma(gru_latent) + self.gru_film_beta(gru_latent)
        torso = torch.tanh(self.torso_dense1(torso))
        torso = torch.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.policy_logstd.expand_as(mean), carry


    def forward_sequence(self, obs, dones, carry):
        means = []
        for step in range(obs.shape[0]):
            if step > 0:
                carry = carry * (1 - dones[step - 1].unsqueeze(-1))
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
