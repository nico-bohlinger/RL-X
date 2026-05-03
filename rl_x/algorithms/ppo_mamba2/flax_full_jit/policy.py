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
                    mamba_obs_combine_method=config.algorithm.mamba_obs_combine_method,
                    share_mamba_obs_encoder=config.algorithm.share_mamba_obs_encoder,
                    mamba_d_model=config.algorithm.mamba_d_model,
                    mamba_num_layers=config.algorithm.mamba_num_layers,
                    mamba_expand=config.algorithm.mamba_expand,
                    mamba_state_dim=config.algorithm.mamba_state_dim,
                    mamba_conv_kernel=config.algorithm.mamba_conv_kernel,
                    mamba_down_projection_dim=config.algorithm.mamba_down_projection_dim,
                    mamba_layer_norm_eps=config.algorithm.mamba_layer_norm_eps,
                    mamba_dt_min=config.algorithm.mamba_dt_min,
                    mamba_dt_max=config.algorithm.mamba_dt_max,
                    policy_observation_indices=policy_observation_indices,
                ),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


class Mamba2Block(nn.Module):
    """Small, full-JAX selective SSM block inspired by Mamba-2.

    This is intentionally dependency-free. It keeps the useful PPO property that
    action selection is recurrent and O(1) per environment step, while PPO updates
    can replay contiguous rollout sequences from their saved initial recurrent state.

    Carry shapes for batched one-step calls:
      ssm:  [B, inner_dim, state_dim]
      conv: [B, conv_kernel - 1, inner_dim]
    """

    d_model: int
    state_dim: int
    expand: int
    conv_kernel: int
    layer_norm_eps: float
    dt_min: float
    dt_max: float

    def setup(self):
        self.inner_dim = int(self.d_model * self.expand)

        self.norm = nn.LayerNorm(epsilon=self.layer_norm_eps)
        self.in_proj = nn.Dense(2 * self.inner_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.x_proj = nn.Dense(self.inner_dim + 2 * self.state_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.out_proj = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        # Depthwise causal convolution parameters. Shape [K, C].
        self.conv_kernel_param = self.param(
            "conv_kernel",
            nn.initializers.normal(stddev=0.02),
            (self.conv_kernel, self.inner_dim),
        )
        self.conv_bias = self.param("conv_bias", constant(0.0), (self.inner_dim,))

        # Diagonal SSM parameters. A is negative for stability.
        a_init = jnp.log(jnp.arange(1, self.state_dim + 1, dtype=jnp.float32))[None, :]
        a_init = jnp.tile(a_init, (self.inner_dim, 1))
        self.A_log = self.param("A_log", lambda key, shape: a_init, (self.inner_dim, self.state_dim))
        self.D = self.param("D", constant(1.0), (self.inner_dim,))

        # Bias dt toward a reasonable initial range. The random draw must live
        # inside the parameter initializer so apply() does not require a params RNG.
        def dt_bias_init(key, shape):
            dt = jnp.exp(
                jax.random.uniform(
                    key,
                    shape,
                    minval=jnp.log(self.dt_min),
                    maxval=jnp.log(self.dt_max),
                )
            )
            return dt + jnp.log(-jnp.expm1(-dt))

        self.dt_bias = self.param("dt_bias", dt_bias_init, (self.inner_dim,))

    def _causal_conv_one_step(self, x, conv_state):
        """Depthwise causal conv for one token.

        x:          [B, inner_dim]
        conv_state: [B, K-1, inner_dim]
        """
        if self.conv_kernel <= 1:
            conv_in = x[:, None, :]
            next_conv_state = conv_state
        else:
            conv_in = jnp.concatenate([conv_state, x[:, None, :]], axis=1)  # [B,K,C]
            next_conv_state = conv_in[:, 1:, :]

        y = jnp.sum(conv_in * self.conv_kernel_param[None, :, :], axis=1) + self.conv_bias[None, :]
        return y, next_conv_state

    def _ssm_one_step(self, u, ssm_state):
        """Selective diagonal SSM update.

        u:         [B, inner_dim]
        ssm_state: [B, inner_dim, state_dim]
        """
        params = self.x_proj(u)
        dt_raw = params[..., :self.inner_dim]
        b_t = params[..., self.inner_dim:self.inner_dim + self.state_dim]
        c_t = params[..., self.inner_dim + self.state_dim:]

        dt = nn.softplus(dt_raw + self.dt_bias[None, :])  # [B, inner_dim]
        A = -jnp.exp(self.A_log)                          # [inner_dim, state_dim]

        dA = jnp.exp(dt[..., None] * A[None, :, :])        # [B, inner_dim, state_dim]
        dB_u = dt[..., None] * b_t[:, None, :] * u[..., None]
        next_ssm_state = dA * ssm_state + dB_u

        y = jnp.sum(next_ssm_state * c_t[:, None, :], axis=-1) + self.D[None, :] * u
        return y, next_ssm_state

    def apply_one_step(self, x, carry):
        """x: [B, d_model]."""
        residual = x
        x = self.norm(x)

        projected = self.in_proj(x)
        u, z = jnp.split(projected, 2, axis=-1)

        u, next_conv_state = self._causal_conv_one_step(u, carry["conv"])
        u = nn.silu(u)

        y, next_ssm_state = self._ssm_one_step(u, carry["ssm"])
        y = y * nn.silu(z)
        y = self.out_proj(y)

        next_carry = {
            "ssm": next_ssm_state,
            "conv": next_conv_state,
        }
        return residual + y, next_carry

    def forward_sequence(self, x_seq, done_seq, init_carry):
        """Replay one environment sequence.

        x_seq:      [T, d_model]
        done_seq:   [T] done after step t
        init_carry: {"ssm":[inner_dim,state_dim], "conv":[K-1,inner_dim]}
        """
        done_prev = jnp.concatenate(
            [jnp.zeros((1,), dtype=jnp.float32), done_seq.astype(jnp.float32)[:-1]],
            axis=0,
        )

        def step(carry, inp):
            x_t, done_prev_t = inp
            reset = 1.0 - done_prev_t
            carry = {
                "ssm": carry["ssm"] * reset,
                "conv": carry["conv"] * reset,
            }
            batched_carry = {
                "ssm": carry["ssm"][None, ...],
                "conv": carry["conv"][None, ...],
            }
            y_t, next_batched_carry = self.apply_one_step(x_t[None, :], batched_carry)
            next_carry = {
                "ssm": next_batched_carry["ssm"][0],
                "conv": next_batched_carry["conv"][0],
            }
            return next_carry, y_t[0]

        _, y_seq = jax.lax.scan(step, init_carry, (x_seq, done_prev), unroll=True)
        return y_seq


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    mamba_obs_combine_method: str
    share_mamba_obs_encoder: bool
    mamba_d_model: int
    mamba_num_layers: int
    mamba_expand: int
    mamba_state_dim: int
    mamba_conv_kernel: int
    mamba_down_projection_dim: int
    mamba_layer_norm_eps: float
    mamba_dt_min: float
    mamba_dt_max: float
    policy_observation_indices: Sequence[int]

    def setup(self):
        act_dim = int(np.prod(self.as_shape).item())
        self.inner_dim = int(self.mamba_d_model * self.mamba_expand)

        self.mamba_obs_enc_dense = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.mamba_obs_enc_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)

        if not self.share_mamba_obs_encoder:
            self.obs_enc_dense = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_enc_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)

        self.mamba_layers = [
            Mamba2Block(
                d_model=self.mamba_d_model,
                state_dim=self.mamba_state_dim,
                expand=self.mamba_expand,
                conv_kernel=self.mamba_conv_kernel,
                layer_norm_eps=self.mamba_layer_norm_eps,
                dt_min=self.mamba_dt_min,
                dt_max=self.mamba_dt_max,
                name=f"mamba2_block_{i}",
            )
            for i in range(self.mamba_num_layers)
        ]
        self.mamba_out_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)
        self.mamba_down_projection_dense = nn.Dense(self.mamba_down_projection_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        if self.mamba_obs_combine_method == "film":
            self.mamba_film_gamma = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.mamba_film_beta = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.torso_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_ln1 = nn.LayerNorm()
        self.torso_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense3 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.mean_head = nn.Dense(act_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, act_dim))

    def initialize_carry(self, batch_size: int):
        # This method is intentionally independent of setup(), because the
        # algorithm calls it before policy.init().
        inner_dim = int(self.mamba_d_model * self.mamba_expand)
        conv_len = max(self.mamba_conv_kernel - 1, 0)
        return {
            "ssm": jnp.zeros((batch_size, self.mamba_num_layers, inner_dim, self.mamba_state_dim), dtype=jnp.float32),
            "conv": jnp.zeros((batch_size, self.mamba_num_layers, conv_len, inner_dim), dtype=jnp.float32),
        }

    def obs_encode(self, obs):
        if self.share_mamba_obs_encoder:
            return self.mamba_obs_encode(obs)
        x = obs[..., self.policy_observation_indices]
        x = self.obs_enc_dense(x)
        x = self.obs_enc_ln(x)
        x = nn.elu(x)
        return x

    def mamba_obs_encode(self, obs):
        x = obs[..., self.policy_observation_indices]
        x = self.mamba_obs_enc_dense(x)
        x = self.mamba_obs_enc_ln(x)
        x = nn.elu(x)
        return x

    def decode(self, obs_latent, mamba_latent):
        mamba_latent = self.mamba_out_ln(mamba_latent)
        mamba_latent = nn.elu(mamba_latent)
        mamba_latent_proj = self.mamba_down_projection_dense(mamba_latent)
        mamba_latent_proj = nn.elu(mamba_latent_proj)

        if self.mamba_obs_combine_method == "concat":
            torso_in = jnp.concatenate([obs_latent, mamba_latent_proj], axis=-1)
        elif self.mamba_obs_combine_method == "film":
            gamma = self.mamba_film_gamma(mamba_latent_proj)
            beta = self.mamba_film_beta(mamba_latent_proj)
            torso_in = obs_latent * gamma + beta
        else:
            raise ValueError(f"Unknown mamba_obs_combine_method: {self.mamba_obs_combine_method}")

        h = self.torso_dense1(torso_in)
        h = self.torso_ln1(h)
        h = nn.elu(h)
        h = self.torso_dense2(h)
        h = nn.elu(h)
        h = self.torso_dense3(h)
        h = nn.elu(h)

        mean = self.mean_head(h)
        return mean, self.logstd

    def apply_one_step(self, obs, carry):
        x = self.mamba_obs_encode(obs)

        next_ssm = []
        next_conv = []
        for i, layer in enumerate(self.mamba_layers):
            layer_carry = {
                "ssm": carry["ssm"][:, i, :, :],
                "conv": carry["conv"][:, i, :, :],
            }
            x, next_layer_carry = layer.apply_one_step(x, layer_carry)
            next_ssm.append(next_layer_carry["ssm"])
            next_conv.append(next_layer_carry["conv"])

        next_carry = {
            "ssm": jnp.stack(next_ssm, axis=1),
            "conv": jnp.stack(next_conv, axis=1),
        }

        obs_latent = self.mamba_obs_encode(obs) if self.share_mamba_obs_encoder else self.obs_encode(obs)
        mean, log_std = self.decode(obs_latent, x)
        return mean, log_std, next_carry

    def forward_sequence(self, obs_seq, done_seq, init_carry):
        """
        obs_seq:    [T, obs_dim]
        done_seq:   [T] done after step t
        init_carry: carry valid for obs_seq[0]
                    {"ssm":[num_layers,inner_dim,state_dim],
                     "conv":[num_layers,K-1,inner_dim]}
        """
        x = self.mamba_obs_encode(obs_seq)

        for i, layer in enumerate(self.mamba_layers):
            layer_carry = {
                "ssm": init_carry["ssm"][i, :, :],
                "conv": init_carry["conv"][i, :, :],
            }
            x = layer.forward_sequence(x, done_seq, layer_carry)

        obs_latent = self.mamba_obs_encode(obs_seq) if self.share_mamba_obs_encoder else self.obs_encode(obs_seq)
        mean, log_std = self.decode(obs_latent, x)
        return mean, log_std


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
