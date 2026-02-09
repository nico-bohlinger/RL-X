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
        return (Policy(
                    as_shape=env.single_action_space.shape,
                    std_dev=config.algorithm.std_dev,
                    tf_obs_combine_method=config.algorithm.tf_obs_combine_method,
                    share_tf_obs_encoder=config.algorithm.share_tf_obs_encoder,
                    tf_d_model=config.algorithm.tf_d_model,
                    tf_dim_feedforward=config.algorithm.tf_dim_feedforward,
                    tf_down_projection_dim=config.algorithm.tf_down_projection_dim,
                    tf_nhead=config.algorithm.tf_nhead,
                    tf_num_layers=config.algorithm.tf_num_layers,
                    tf_dropout=config.algorithm.tf_dropout,
                    tf_layer_norm_eps=config.algorithm.tf_layer_norm_eps,
                    tf_context_len=config.algorithm.tf_context_len,
                    policy_observation_indices=policy_observation_indices,
                ),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


def _sinusoidal_positional_encoding(length: int, d_model: int, dtype=jnp.float32) -> jnp.ndarray:
    positions = jnp.arange(length, dtype=dtype)[:, None]  # [L,1]
    div = jnp.exp(
        (jnp.arange(0, d_model, 2, dtype=dtype) * (-jnp.log(jnp.array(10000.0, dtype=dtype)) / d_model))
    )  # [d_model/2]

    pe_even = jnp.sin(positions * div[None, :])  # [L, d_model/2]
    pe_odd = jnp.cos(positions * div[None, :])   # [L, d_model/2]

    pe = jnp.zeros((length, d_model), dtype=dtype)
    pe = pe.at[:, 0::2].set(pe_even)
    if d_model % 2 == 0:
        pe = pe.at[:, 1::2].set(pe_odd)
    else:
        pe = pe.at[:, 1::2].set(pe_odd[:, : (d_model // 2)])
    return pe


class TransformerEncoderLayer(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    layer_norm_eps: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                      # [B,T,d_model]
        *,
        padding_mask: jnp.ndarray | None,    # [B,T] bool (True = token present)
        attn_mask: jnp.ndarray | None,       # [T,T] or [B,T,T] bool (True = keep)
        deterministic: bool,
        is_causal: bool,
    ) -> jnp.ndarray:
        ln1 = nn.LayerNorm(epsilon=self.layer_norm_eps)
        ln2 = nn.LayerNorm(epsilon=self.layer_norm_eps)

        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            dropout_rate=self.dropout,
            broadcast_dropout=False,
        )

        do_attn = nn.Dropout(rate=self.dropout)
        do_ff1 = nn.Dropout(rate=self.dropout)
        do_ff2 = nn.Dropout(rate=self.dropout)

        keep = None  # bool mask broadcastable to [B,1,T,T]

        if padding_mask is not None:
            pm = padding_mask.astype(bool)  # [B,T]
            keep_pm = pm[:, None, :, None] & pm[:, None, None, :]  # [B,1,T,T]
            keep = keep_pm if keep is None else (keep & keep_pm)

        if attn_mask is not None:
            if attn_mask.ndim == 2:            # [T,T]
                am = attn_mask[None, None, :, :]   # [1,1,T,T]
            elif attn_mask.ndim == 3:          # [B,T,T]
                am = attn_mask[:, None, :, :]      # [B,1,T,T]
            else:
                am = attn_mask
            keep = am if keep is None else (keep & am)

        if is_causal:
            T = x.shape[-2]
            causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
            keep = causal if keep is None else (keep & causal)

        # Pre-norm self-attention
        y = ln1(x)
        sa = mha(y, mask=keep, deterministic=deterministic)  # [B,T,d_model]
        sa = do_attn(sa, deterministic=deterministic)
        x = x + sa

        # Pre-norm FFN
        y = ln2(x)
        y = nn.Dense(self.dim_feedforward)(y)
        y = nn.relu(y)
        y = do_ff1(y, deterministic=deterministic)
        y = nn.Dense(self.d_model)(y)
        y = do_ff2(y, deterministic=deterministic)
        x = x + y

        return x


class TransformerEncoder(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float
    num_layers: int
    layer_norm_eps: float

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,                      # [B,T,d_model]
        *,
        padding_mask: jnp.ndarray | None,    # [B,T] bool
        attn_mask: jnp.ndarray | None,       # bool keep mask
        deterministic: bool,
        is_causal: bool,
    ) -> jnp.ndarray:
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                layer_norm_eps=self.layer_norm_eps,
            )(
                x,
                padding_mask=padding_mask,
                attn_mask=attn_mask,
                deterministic=deterministic,
                is_causal=is_causal,
            )
        return x


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    tf_obs_combine_method: str
    share_tf_obs_encoder: bool
    tf_d_model: int
    tf_dim_feedforward: int
    tf_down_projection_dim: int
    tf_down_projection_dim: int
    tf_nhead: int
    tf_num_layers: int
    tf_dropout: float
    tf_layer_norm_eps: float
    tf_context_len: int
    policy_observation_indices: Sequence[int]

    def setup(self):
        act_dim = int(np.prod(self.as_shape).item())

        self.tf_obs_enc_dense1 = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.tf_obs_enc_ln1 = nn.LayerNorm()

        if not self.share_tf_obs_encoder:
            self.obs_enc_dense1 = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_enc_ln1 = nn.LayerNorm()
        
        self.tf_down_projection_dense = nn.Dense(self.tf_down_projection_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        if self.tf_obs_combine_method == "film":
            self.tf_film_gamma = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.tf_film_beta = nn.Dense(self.tf_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        
        self.transformer = TransformerEncoder(
            d_model=self.tf_d_model,
            nhead=self.tf_nhead,
            dim_feedforward=self.tf_dim_feedforward,
            dropout=self.tf_dropout,
            num_layers=self.tf_num_layers,
            layer_norm_eps=self.tf_layer_norm_eps,
        )

        self.torso_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_ln1 = nn.LayerNorm()
        self.torso_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense3 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.mean_head = nn.Dense(act_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, act_dim))


    def initialize_history(self, batch_size: int, obs_dim: int):
        obs_hist = jnp.zeros((batch_size, self.tf_context_len - 1, obs_dim), dtype=jnp.float32)
        mask_hist = jnp.zeros((batch_size, self.tf_context_len - 1), dtype=bool)
        return {"obs": obs_hist, "mask": mask_hist}


    def _reset_history(self, history):
        return {"obs": jnp.zeros_like(history["obs"]), "mask": jnp.zeros_like(history["mask"])}


    def _update_history(self, history, obs):
        if history["obs"].shape[1] == 0:
            return history
        new_obs = jnp.concatenate([history["obs"][:, 1:, :], obs[:, None, :]], axis=1)
        new_mask = jnp.concatenate([history["mask"][:, 1:], jnp.ones((history["mask"].shape[0], 1), dtype=bool)], axis=1)
        return {"obs": new_obs, "mask": new_mask}


    def obs_encode(self, obs):
        if self.share_tf_obs_encoder:
            return self.tf_obs_encode(obs)
        x = obs[..., self.policy_observation_indices]
        x = self.obs_enc_dense1(x)
        x = self.obs_enc_ln1(x)
        x = nn.elu(x)
        return x


    def tf_obs_encode(self, obs):
        x = obs[..., self.policy_observation_indices]
        x = self.tf_obs_enc_dense1(x)
        x = self.tf_obs_enc_ln1(x)
        x = nn.elu(x)
        return x


    def decode(self, obs_latent, tf_latent):
        tf_latent_proj = self.tf_down_projection_dense(tf_latent)
        tf_latent_proj = nn.elu(tf_latent_proj)

        if self.tf_obs_combine_method == "concat":
            torso_in = jnp.concatenate([obs_latent, tf_latent_proj], axis=-1)
        elif self.tf_obs_combine_method == "film":
            gamma = self.tf_film_gamma(tf_latent_proj)
            beta = self.tf_film_beta(tf_latent_proj)
            torso_in = obs_latent * gamma + beta

        h = self.torso_dense1(torso_in)
        h = self.torso_ln1(h)
        h = nn.elu(h)
        h = self.torso_dense2(h)
        h = nn.elu(h)
        h = self.torso_dense3(h)
        h = nn.elu(h)

        mean = self.mean_head(h)
        
        return mean, self.logstd
    

    def apply_one_step(self, obs, history):
        """
        obs:     [B, obs_dim]
        history: {"obs":[B,L,obs_dim], "mask":[B,L]} where L=tf_context_len-1
        """
        B = obs.shape[0]
        L = history["obs"].shape[1]

        # token seq: [B, T=L+1, obs_dim]
        obs_seq = jnp.concatenate([history["obs"], obs[:, None, :]], axis=1) if L > 0 else obs[:, None, :]
        pad = jnp.concatenate([history["mask"], jnp.ones((B, 1), dtype=bool)], axis=1) if L > 0 else jnp.ones((B, 1), dtype=bool)

        x = self.tf_obs_encode(obs_seq)  # [B,T,d_model]
        T = x.shape[1]
        x = x + _sinusoidal_positional_encoding(T, self.tf_d_model, dtype=x.dtype)[None, :, :]

        h = self.transformer(x, padding_mask=pad, attn_mask=None, deterministic=True, is_causal=True)  # [B,T,d_model]

        tf_last = h[:, -1, :]              # [B,d_model]
        obs_lat = self.obs_encode(obs)     # [B,d_model]

        mean, log_std = self.decode(obs_lat, tf_last)

        next_history = self._update_history(history, obs)
        return mean, log_std, next_history


    def forward_sequence(self, obs_seq, done_seq, init_history):
        """
        obs_seq:      [T, obs_dim]
        done_seq:     [T] (done after step t)
        init_history: {"obs":[L,obs_dim], "mask":[L]} context valid for obs_seq[0]
                     where L=tf_context_len-1

        Returns:
          mean_seq:   [T, act_dim]
          logstd_seq: [T, act_dim]
        """
        T = obs_seq.shape[0]
        L = init_history["obs"].shape[0]

        # Build extended token sequence: [L+T, obs_dim]
        if L > 0:
            obs_ext = jnp.concatenate([init_history["obs"], obs_seq], axis=0)
            pad_ext = jnp.concatenate([init_history["mask"], jnp.ones((T,), dtype=bool)], axis=0)  # [L+T]
        else:
            obs_ext = obs_seq
            pad_ext = jnp.ones((T,), dtype=bool)

        T_ext = obs_ext.shape[0]

        # Reset BEFORE obs_seq[t] if done at t-1
        done_prev_obs = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), done_seq.astype(jnp.float32)[:-1]], axis=0)  # [T]
        if L > 0:
            done_prev_ext = jnp.concatenate([jnp.zeros((L,), dtype=jnp.float32), done_prev_obs], axis=0)  # [L+T]
        else:
            done_prev_ext = done_prev_obs  # [T]

        seg = jnp.cumsum(done_prev_ext.astype(jnp.int32), axis=0)  # [T_ext]
        same_seg = (seg[:, None] == seg[None, :])  # [T_ext,T_ext]

        idx = jnp.arange(T_ext, dtype=jnp.int32)
        causal = (idx[:, None] >= idx[None, :])
        within = (idx[:, None] - idx[None, :]) < self.tf_context_len
        band = causal & within
        attn_mask = band & same_seg  # [T_ext,T_ext], bool keep

        x = self.tf_obs_encode(obs_ext)[None, ...]  # [1,T_ext,d_model]
        x = x + _sinusoidal_positional_encoding(T_ext, self.tf_d_model, dtype=x.dtype)[None, :, :]

        h = self.transformer(x, padding_mask=pad_ext[None, :], attn_mask=attn_mask, deterministic=True, is_causal=False)[0]  # [T_ext,d_model]

        tf_lat = h[L:, :]  # [T,d_model] aligned with obs_seq
        obs_lat = self.obs_encode(obs_seq)  # [T,d_model]

        mean, log_std = self.decode(obs_lat, tf_lat)
        return mean, log_std


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
