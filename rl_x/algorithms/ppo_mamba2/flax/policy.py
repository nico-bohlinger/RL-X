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
        return (Policy(env.single_action_space.shape, config.algorithm.std_dev, config.algorithm.mamba_obs_combine_method, config.algorithm.share_mamba_obs_encoder, config.algorithm.mamba_d_model, config.algorithm.mamba_num_layers, config.algorithm.mamba_expand, config.algorithm.mamba_state_dim, config.algorithm.mamba_conv_kernel, config.algorithm.mamba_down_projection_dim, config.algorithm.mamba_layer_norm_eps, config.algorithm.mamba_dt_min, config.algorithm.mamba_dt_max, config.algorithm.nr_hidden_units, policy_observation_indices),
                get_processed_action_function(config.algorithm.action_clipping_and_rescaling, jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)))


class Mamba2Block(nn.Module):
    d_model: int
    state_dim: int
    expand: int
    conv_kernel: int
    layer_norm_eps: float
    dt_min: float
    dt_max: float

    def setup(self):
        self.inner_dim = self.d_model * self.expand
        self.norm = nn.LayerNorm(epsilon=self.layer_norm_eps)
        self.in_proj = nn.Dense(2 * self.inner_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.x_proj = nn.Dense(self.inner_dim + 2 * self.state_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.out_proj = nn.Dense(self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.conv_kernel_param = self.param("conv_kernel", nn.initializers.normal(stddev=0.02), (self.conv_kernel, self.inner_dim))
        self.conv_bias = self.param("conv_bias", constant(0.0), (self.inner_dim,))
        a_init = jnp.tile(jnp.log(jnp.arange(1, self.state_dim + 1, dtype=jnp.float32))[None], (self.inner_dim, 1))
        self.A_log = self.param("A_log", lambda key, shape: a_init, (self.inner_dim, self.state_dim))
        self.D = self.param("D", constant(1.0), (self.inner_dim,))

        def dt_bias_init(key, shape):
            dt = jnp.exp(jax.random.uniform(key, shape, minval=jnp.log(self.dt_min), maxval=jnp.log(self.dt_max)))
            return dt + jnp.log(-jnp.expm1(-dt))

        self.dt_bias = self.param("dt_bias", dt_bias_init, (self.inner_dim,))


    def causal_conv_one_step(self, x, conv_state):
        if self.conv_kernel <= 1:
            conv_input = x[:, None]
            next_conv_state = conv_state
        else:
            conv_input = jnp.concatenate([conv_state, x[:, None]], axis=1)
            next_conv_state = conv_input[:, 1:]
        return jnp.sum(conv_input * self.conv_kernel_param[None], axis=1) + self.conv_bias[None], next_conv_state


    def ssm_one_step(self, u, ssm_state):
        params = self.x_proj(u)
        dt_raw = params[..., :self.inner_dim]
        b_t = params[..., self.inner_dim:self.inner_dim + self.state_dim]
        c_t = params[..., self.inner_dim + self.state_dim:]
        dt = nn.softplus(dt_raw + self.dt_bias[None])
        dA = jnp.exp(dt[..., None] * -jnp.exp(self.A_log)[None])
        next_ssm_state = dA * ssm_state + dt[..., None] * b_t[:, None] * u[..., None]
        return jnp.sum(next_ssm_state * c_t[:, None], axis=-1) + self.D[None] * u, next_ssm_state


    def apply_one_step(self, x, carry):
        residual = x
        u, z = jnp.split(self.in_proj(self.norm(x)), 2, axis=-1)
        u, next_conv_state = self.causal_conv_one_step(u, carry["conv"])
        y, next_ssm_state = self.ssm_one_step(nn.silu(u), carry["ssm"])
        return residual + self.out_proj(y * nn.silu(z)), {"ssm": next_ssm_state, "conv": next_conv_state}


    def forward_sequence(self, x_sequence, done_sequence, init_carry):
        done_previous = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), done_sequence[:-1].astype(jnp.float32)])

        def step(carry, inputs):
            x, done = inputs
            carry = {"ssm": carry["ssm"] * (1 - done), "conv": carry["conv"] * (1 - done)}
            x, carry = self.apply_one_step(x[None], {key: value[None] for key, value in carry.items()})
            return {key: value[0] for key, value in carry.items()}, x[0]

        return jax.lax.scan(step, init_carry, (x_sequence, done_previous), unroll=True)[1]


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
    nr_hidden_units: int
    policy_observation_indices: Sequence[int]

    def setup(self):
        action_dim = np.prod(self.as_shape).item()
        self.mamba_obs_encoder_dense = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.mamba_obs_encoder_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)
        if not self.share_mamba_obs_encoder:
            self.obs_encoder_dense = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_encoder_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)
        self.mamba_layers = [Mamba2Block(self.mamba_d_model, self.mamba_state_dim, self.mamba_expand, self.mamba_conv_kernel, self.mamba_layer_norm_eps, self.mamba_dt_min, self.mamba_dt_max, name=f"mamba2_block_{index}") for index in range(self.mamba_num_layers)]
        self.mamba_out_ln = nn.LayerNorm(epsilon=self.mamba_layer_norm_eps)
        self.mamba_down_projection = nn.Dense(self.mamba_down_projection_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        if self.mamba_obs_combine_method == "film":
            self.mamba_film_gamma = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.mamba_film_beta = nn.Dense(self.mamba_d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense1 = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense2 = nn.Dense(self.nr_hidden_units, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.mean_head = nn.Dense(action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, action_dim))


    def initialize_carry(self, batch_size):
        inner_dim = self.mamba_d_model * self.mamba_expand
        return {
            "ssm": jnp.zeros((batch_size, self.mamba_num_layers, inner_dim, self.mamba_state_dim), dtype=jnp.float32),
            "conv": jnp.zeros((batch_size, self.mamba_num_layers, max(self.mamba_conv_kernel - 1, 0), inner_dim), dtype=jnp.float32),
        }


    def mamba_obs_encode(self, obs):
        obs = obs[..., self.policy_observation_indices]
        return nn.elu(self.mamba_obs_encoder_ln(self.mamba_obs_encoder_dense(obs)))


    def obs_encode(self, obs):
        if self.share_mamba_obs_encoder:
            return self.mamba_obs_encode(obs)
        obs = obs[..., self.policy_observation_indices]
        return nn.elu(self.obs_encoder_ln(self.obs_encoder_dense(obs)))


    def decode(self, obs_latent, mamba_latent):
        mamba_latent = nn.elu(self.mamba_out_ln(mamba_latent))
        mamba_latent = nn.elu(self.mamba_down_projection(mamba_latent))
        if self.mamba_obs_combine_method == "concat":
            torso = jnp.concatenate([obs_latent, mamba_latent], axis=-1)
        else:
            torso = obs_latent * self.mamba_film_gamma(mamba_latent) + self.mamba_film_beta(mamba_latent)
        torso = nn.tanh(self.torso_dense1(torso))
        torso = nn.tanh(self.torso_dense2(torso))
        mean = self.mean_head(torso)
        return mean, self.logstd


    def apply_one_step(self, obs, carry):
        x = self.mamba_obs_encode(obs)
        next_ssm = []
        next_conv = []
        for index, layer in enumerate(self.mamba_layers):
            x, next_carry = layer.apply_one_step(x, {"ssm": carry["ssm"][:, index], "conv": carry["conv"][:, index]})
            next_ssm.append(next_carry["ssm"])
            next_conv.append(next_carry["conv"])
        next_carry = {"ssm": jnp.stack(next_ssm, axis=1), "conv": jnp.stack(next_conv, axis=1)}
        mean, logstd = self.decode(self.obs_encode(obs), x)
        return mean, logstd, next_carry


    def forward_sequence(self, obs_sequence, done_sequence, init_carry):
        x = self.mamba_obs_encode(obs_sequence)
        for index, layer in enumerate(self.mamba_layers):
            x = layer.forward_sequence(x, done_sequence, {"ssm": init_carry["ssm"][index], "conv": init_carry["conv"][index]})
        return self.decode(self.obs_encode(obs_sequence), x)


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + 0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low)
        return jax.jit(get_clipped_and_scaled_action)
    return jax.jit(lambda x: x)
