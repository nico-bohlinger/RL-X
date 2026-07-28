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
        policy = Policy(env, config.algorithm.std_dev, config.algorithm.action_clipping_and_rescaling, config.algorithm.tf_obs_combine_method, config.algorithm.share_tf_obs_encoder, config.algorithm.tf_d_model, config.algorithm.tf_dim_feedforward, config.algorithm.tf_down_projection_dim, config.algorithm.tf_nhead, config.algorithm.tf_num_layers, config.algorithm.tf_dropout, config.algorithm.tf_layer_norm_eps, config.algorithm.tf_context_len, config.algorithm.nr_hidden_units, device, policy_observation_indices).to(device)

    policy.get_action_logprob = torch.compile(policy.get_action_logprob, mode=config.algorithm.compile_mode)
    policy.get_logprob_entropy = torch.compile(policy.get_logprob_entropy, mode=config.algorithm.compile_mode)
    policy.get_deterministic_action = torch.compile(policy.get_deterministic_action, mode=config.algorithm.compile_mode)
    return policy


def sinusoidal_positional_encoding(length, d_model, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / d_model))
    encoding = torch.zeros((length, d_model), device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div[None])
    encoding[:, 1::2] = torch.cos(positions * div[None])[:, :d_model // 2]
    return encoding


class Policy(nn.Module):
    def __init__(self, env, std_dev, action_clipping_and_rescaling, tf_obs_combine_method, share_tf_obs_encoder, tf_d_model, tf_dim_feedforward, tf_down_projection_dim, tf_nhead, tf_num_layers, tf_dropout, tf_layer_norm_eps, tf_context_len, nr_hidden_units, device, policy_observation_indices):
        super().__init__()
        self.action_clipping_and_rescaling = action_clipping_and_rescaling
        self.tf_obs_combine_method = tf_obs_combine_method
        self.share_tf_obs_encoder = share_tf_obs_encoder
        self.tf_d_model = tf_d_model
        self.tf_nhead = tf_nhead
        self.tf_context_len = tf_context_len
        self.policy_observation_indices = torch.tensor(policy_observation_indices, dtype=torch.long, device=device)
        self.env_as_low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=device)
        self.env_as_high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=device)
        action_dim = np.prod(env.single_action_space.shape, dtype=int).item()

        self.tf_obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), tf_d_model))
        self.tf_obs_encoder_ln = nn.LayerNorm(tf_d_model, eps=1e-6)
        if not share_tf_obs_encoder:
            self.obs_encoder_dense = self.layer_init(nn.Linear(len(policy_observation_indices), tf_d_model))
            self.obs_encoder_ln = nn.LayerNorm(tf_d_model, eps=1e-6)
        self.tf_down_projection = self.layer_init(nn.Linear(tf_d_model, tf_down_projection_dim))
        if tf_obs_combine_method == "film":
            self.tf_film_gamma = self.layer_init(nn.Linear(tf_down_projection_dim, tf_d_model))
            self.tf_film_beta = self.layer_init(nn.Linear(tf_down_projection_dim, tf_d_model))
        encoder_layer = nn.TransformerEncoderLayer(tf_d_model, tf_nhead, tf_dim_feedforward, tf_dropout, activation="relu", layer_norm_eps=tf_layer_norm_eps, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, tf_num_layers, enable_nested_tensor=False)
        torso_input_dim = tf_d_model + tf_down_projection_dim if tf_obs_combine_method == "concat" else tf_d_model
        self.torso_dense1 = self.layer_init(nn.Linear(torso_input_dim, nr_hidden_units))
        self.torso_dense2 = self.layer_init(nn.Linear(nr_hidden_units, nr_hidden_units))
        self.mean_head = self.layer_init(nn.Linear(nr_hidden_units, action_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.full((1, action_dim), np.log(std_dev).item()))


    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    def initialize_history(self, batch_size, obs_dim):
        return {"obs": torch.zeros((batch_size, self.tf_context_len - 1, obs_dim), dtype=torch.float32, device=self.policy_logstd.device), "mask": torch.zeros((batch_size, self.tf_context_len - 1), dtype=torch.bool, device=self.policy_logstd.device)}


    def update_history(self, history, obs):
        if history["obs"].shape[1] == 0:
            return history
        return {"obs": torch.cat([history["obs"][:, 1:], obs[:, None]], dim=1), "mask": torch.cat([history["mask"][:, 1:], torch.ones((history["mask"].shape[0], 1), dtype=torch.bool, device=obs.device)], dim=1)}


    def tf_obs_encode(self, obs):
        obs = obs[..., self.policy_observation_indices]
        return torch.nn.functional.elu(self.tf_obs_encoder_ln(self.tf_obs_encoder_dense(obs)))


    def obs_encode(self, obs):
        if self.share_tf_obs_encoder:
            return self.tf_obs_encode(obs)
        obs = obs[..., self.policy_observation_indices]
        return torch.nn.functional.elu(self.obs_encoder_ln(self.obs_encoder_dense(obs)))


    def decode(self, obs_latent, tf_latent):
        tf_latent = torch.nn.functional.elu(self.tf_down_projection(tf_latent))
        if self.tf_obs_combine_method == "concat":
            torso = torch.cat([obs_latent, tf_latent], dim=-1)
        else:
            torso = obs_latent * self.tf_film_gamma(tf_latent) + self.tf_film_beta(tf_latent)
        torso = torch.tanh(self.torso_dense1(torso))
        torso = torch.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.policy_logstd


    def apply_one_step(self, obs, history):
        obs_sequence = torch.cat([history["obs"], obs[:, None]], dim=1)
        padding_mask = torch.cat([history["mask"], torch.ones((obs.shape[0], 1), dtype=torch.bool, device=obs.device)], dim=1)
        latent = self.tf_obs_encode(obs_sequence)
        latent = latent + sinusoidal_positional_encoding(latent.shape[1], self.tf_d_model, latent.device, latent.dtype)[None]
        attention_mask = torch.triu(torch.ones((latent.shape[1], latent.shape[1]), dtype=torch.bool, device=latent.device), diagonal=1)
        tf_latent = self.transformer(latent, mask=attention_mask, src_key_padding_mask=~padding_mask)[:, -1]
        mean, logstd = self.decode(self.obs_encode(obs), tf_latent)
        return mean, logstd.expand_as(mean), self.update_history(history, obs)


    def forward_sequence(self, obs_sequence, done_sequence, init_history):
        batch_size = obs_sequence.shape[1]
        history_length = init_history["obs"].shape[1]
        obs_extended = torch.cat([init_history["obs"], obs_sequence.transpose(0, 1)], dim=1)
        padding_mask = torch.cat([init_history["mask"], torch.ones((batch_size, obs_sequence.shape[0]), dtype=torch.bool, device=obs_sequence.device)], dim=1)
        done_previous = torch.cat([torch.zeros((batch_size, history_length + 1), dtype=torch.float32, device=obs_sequence.device), done_sequence[:-1].transpose(0, 1)], dim=1)
        segments = torch.cumsum(done_previous.to(torch.int32), dim=1)
        indices = torch.arange(obs_extended.shape[1], device=obs_sequence.device)
        attention_mask = (indices[:, None] >= indices[None]) & ((indices[:, None] - indices[None]) < self.tf_context_len)
        attention_mask = attention_mask[None] & (segments[:, :, None] == segments[:, None, :])
        attention_mask = (~attention_mask).repeat_interleave(self.tf_nhead, dim=0)
        latent = self.tf_obs_encode(obs_extended)
        latent = latent + sinusoidal_positional_encoding(latent.shape[1], self.tf_d_model, latent.device, latent.dtype)[None]
        tf_latent = self.transformer(latent, mask=attention_mask, src_key_padding_mask=~padding_mask)[:, history_length:].transpose(0, 1)
        mean, logstd = self.decode(self.obs_encode(obs_sequence), tf_latent)
        return mean, logstd.expand_as(mean)


    def get_action_logprob(self, obs, history):
        mean, logstd, history = self.apply_one_step(obs, history)
        distribution = Normal(mean, logstd.exp())
        action = distribution.sample()
        processed_action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low) if self.action_clipping_and_rescaling else action
        return action, processed_action, distribution.log_prob(action).sum(-1), history


    def get_logprob_entropy(self, obs, dones, history, action):
        mean, logstd = self.forward_sequence(obs, dones, history)
        distribution = Normal(mean, logstd.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1)


    def get_deterministic_action(self, obs, history):
        action, logstd, history = self.apply_one_step(obs, history)
        if self.action_clipping_and_rescaling:
            action = self.env_as_low + 0.5 * (torch.clip(action, -1, 1) + 1) * (self.env_as_high - self.env_as_low)
        return action, history
