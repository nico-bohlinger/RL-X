from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_policy(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type
    policy_observation_indices = getattr(env, "policy_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return (Policy(env.single_action_space.shape, config.algorithm.std_dev, config.algorithm.tf_obs_combine_method, config.algorithm.share_tf_obs_encoder, config.algorithm.tf_d_model, config.algorithm.tf_dim_feedforward, config.algorithm.tf_down_projection_dim, config.algorithm.tf_nhead, config.algorithm.tf_num_layers, config.algorithm.tf_dropout, config.algorithm.tf_layer_norm_eps, config.algorithm.tf_context_len, config.algorithm.nr_hidden_units, policy_observation_indices),
                get_processed_action_function(config.algorithm.action_clipping_and_rescaling, jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))


def sinusoidal_positional_encoding(length, d_model, dtype=jnp.float32):
    positions = jnp.arange(length, dtype=dtype)[:, None]
    div = jnp.exp(jnp.arange(0, d_model, 2, dtype=dtype) * (-jnp.log(jnp.array(10000.0, dtype=dtype)) / d_model))
    encoding = jnp.zeros((length, d_model), dtype=dtype)
    encoding = encoding.at[:, 0::2].set(jnp.sin(positions * div[None, :]))
    encoding = encoding.at[:, 1::2].set(jnp.cos(positions * div[None, :])[:, :d_model // 2])
    return encoding


class TransformerEncoderLayer(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    layer_norm_eps: float

    @nn.compact
    def __call__(self, x, padding_mask, attn_mask, deterministic, is_causal):
        keep = None
        if padding_mask is not None:
            padding_mask = padding_mask.astype(bool)
            keep = padding_mask[:, None, :, None] & padding_mask[:, None, None, :]
        if attn_mask is not None:
            attn_mask = attn_mask[None, None] if attn_mask.ndim == 2 else attn_mask[:, None] if attn_mask.ndim == 3 else attn_mask
            keep = attn_mask if keep is None else keep & attn_mask
        if is_causal:
            causal = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2]), dtype=bool))[None, None]
            keep = causal if keep is None else keep & causal

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.nhead, dropout_rate=self.dropout, broadcast_dropout=False)(y, mask=keep, deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = nn.Dense(self.dim_feedforward)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        y = nn.Dense(self.d_model)(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)
        return x + y


class TransformerEncoder(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_layers: int
    layer_norm_eps: float

    @nn.compact
    def __call__(self, x, padding_mask, attn_mask, deterministic, is_causal):
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.layer_norm_eps)(x, padding_mask, attn_mask, deterministic, is_causal)
        return x


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    tf_obs_combine_method: str
    share_tf_obs_encoder: bool
    tf_d_model: int
    tf_dim_feedforward: int
    tf_down_projection_dim: int
    tf_nhead: int
    tf_num_layers: int
    tf_dropout: float
    tf_layer_norm_eps: float
    tf_context_len: int
    nr_hidden_units: int
    policy_observation_indices: Sequence[int]

    def setup(self):
        action_dim = np.prod(self.as_shape).item()
        self.tf_obs_encoder_dense = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.tf_obs_encoder_ln = nn.LayerNorm()
        if not self.share_tf_obs_encoder:
            self.obs_encoder_dense = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_encoder_ln = nn.LayerNorm()
        self.tf_down_projection = nn.Dense(self.tf_down_projection_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        if self.tf_obs_combine_method == "film":
            self.tf_film_gamma = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.tf_film_beta = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.transformer = TransformerEncoder(self.tf_d_model, self.tf_nhead, self.tf_dim_feedforward, self.tf_dropout, self.tf_num_layers, self.tf_layer_norm_eps)
        self.torso_dense1 = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense2 = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.mean_head = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, action_dim))


    def initialize_history(self, batch_size, obs_dim):
        return {"obs": jnp.zeros((batch_size, self.tf_context_len - 1, obs_dim), dtype=jnp.float32), "mask": jnp.zeros((batch_size, self.tf_context_len - 1), dtype=bool)}


    def update_history(self, history, obs):
        if history["obs"].shape[1] == 0:
            return history
        return {"obs": jnp.concatenate([history["obs"][:, 1:], obs[:, None]], axis=1), "mask": jnp.concatenate([history["mask"][:, 1:], jnp.ones((history["mask"].shape[0], 1), dtype=bool)], axis=1)}


    def tf_obs_encode(self, obs):
        obs = obs[..., self.policy_observation_indices]
        return nn.elu(self.tf_obs_encoder_ln(self.tf_obs_encoder_dense(obs)))


    def obs_encode(self, obs):
        if self.share_tf_obs_encoder:
            return self.tf_obs_encode(obs)
        obs = obs[..., self.policy_observation_indices]
        return nn.elu(self.obs_encoder_ln(self.obs_encoder_dense(obs)))


    def decode(self, obs_latent, tf_latent):
        tf_latent = nn.elu(self.tf_down_projection(tf_latent))
        if self.tf_obs_combine_method == "concat":
            torso = jnp.concatenate([obs_latent, tf_latent], axis=-1)
        else:
            torso = obs_latent * self.tf_film_gamma(tf_latent) + self.tf_film_beta(tf_latent)
        torso = nn.tanh(self.torso_dense1(torso))
        torso = nn.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.logstd


    def apply_one_step(self, obs, history):
        obs_sequence = jnp.concatenate([history["obs"], obs[:, None]], axis=1)
        padding_mask = jnp.concatenate([history["mask"], jnp.ones((obs.shape[0], 1), dtype=bool)], axis=1)
        latent = self.tf_obs_encode(obs_sequence)
        latent = latent + sinusoidal_positional_encoding(latent.shape[1], self.tf_d_model, latent.dtype)[None]
        tf_latent = self.transformer(latent, padding_mask, None, True, True)[:, -1]
        mean, logstd = self.decode(self.obs_encode(obs), tf_latent)
        return mean, logstd, self.update_history(history, obs)


    def forward_sequence(self, obs_sequence, done_sequence, init_history):
        history_length = init_history["obs"].shape[0]
        obs_extended = jnp.concatenate([init_history["obs"], obs_sequence], axis=0)
        padding_mask = jnp.concatenate([init_history["mask"], jnp.ones((obs_sequence.shape[0],), dtype=bool)], axis=0)
        done_previous = jnp.concatenate([jnp.zeros((history_length + 1,), dtype=jnp.float32), done_sequence[:-1]])
        segments = jnp.cumsum(done_previous.astype(jnp.int32))
        indices = jnp.arange(obs_extended.shape[0])
        attention_mask = (indices[:, None] >= indices[None]) & ((indices[:, None] - indices[None]) < self.tf_context_len) & (segments[:, None] == segments[None])
        latent = self.tf_obs_encode(obs_extended)[None]
        latent = latent + sinusoidal_positional_encoding(latent.shape[1], self.tf_d_model, latent.dtype)[None]
        tf_latent = self.transformer(latent, padding_mask[None], attention_mask, True, False)[0, history_length:]
        return self.decode(self.obs_encode(obs_sequence), tf_latent)


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + 0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low)
        return jax.jit(get_clipped_and_scaled_action)
    return jax.jit(lambda x: x)
