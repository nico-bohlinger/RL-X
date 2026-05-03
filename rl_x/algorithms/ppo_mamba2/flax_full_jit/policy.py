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
                    env.single_action_space.shape,
                    config.algorithm.std_dev,
                    config.algorithm.obs_encoding_dim,
                    config.algorithm.gru_hidden_dim,
                    config.algorithm.gru_obs_combine_method,
                    config.algorithm.share_gru_obs_encoder,
                    policy_observation_indices
                ),
                get_processed_action_function(
                    config.algorithm.action_clipping_and_rescaling,
                    jnp.array(env.single_action_space.low), jnp.array(env.single_action_space.high)
                ))


class Policy(nn.Module):
    as_shape: Sequence[int]
    std_dev: float
    obs_encoding_dim: int
    gru_hidden_dim: int
    gru_obs_combine_method: str
    share_gru_obs_encoder: bool
    policy_observation_indices: Sequence[int]

    def setup(self):
        act_dim = int(np.prod(self.as_shape).item())

        self.gru_obs_encoder_dense = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gru_obs_encoder_ln = nn.LayerNorm()

        if not self.share_gru_obs_encoder:
            self.obs_encoder_dense = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_encoder_ln = nn.LayerNorm()

        self.gru = nn.GRUCell(features=self.gru_hidden_dim)
        self.gru_ln = nn.LayerNorm()

        if self.gru_obs_combine_method == "film":
            self.gru_film_gamma = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.gru_film_beta = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.torso_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_ln1 = nn.LayerNorm()
        self.torso_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense3 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.mean_head = nn.Dense(act_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))


    def initialize_carry(self, nr_envs: int):
        return jnp.zeros((nr_envs, self.gru_hidden_dim), dtype=jnp.float32)


    def obs_encode(self, obs):
        x = obs[..., self.policy_observation_indices]
        x = self.obs_encoder_dense(x)
        x = self.obs_encoder_ln(x)
        x = nn.elu(x)
        return x


    def gru_obs_encode(self, obs):
        x = obs[..., self.policy_observation_indices]
        x = self.gru_obs_encoder_dense(x)
        x = self.gru_obs_encoder_ln(x)
        x = nn.elu(x)
        return x


    def decode(self, obs_latent, gru_latent):
        gru_latent = self.gru_ln(gru_latent)
        gru_latent = nn.elu(gru_latent)

        if self.gru_obs_combine_method == "concat":
            torso_in = jnp.concatenate([obs_latent, gru_latent], axis=-1)
        elif self.gru_obs_combine_method == "film":
            gamma = self.gru_film_gamma(gru_latent)
            beta = self.gru_film_beta(gru_latent)
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


    def apply_one_step(self, obs, carry):
        gru_obs_latent = self.gru_obs_encode(obs)
        carry, hidden = self.gru(carry, gru_obs_latent)

        if self.share_gru_obs_encoder:
            obs_latent = gru_obs_latent
        else:
            obs_latent = self.obs_encode(obs)

        mean, log_std = self.decode(obs_latent, hidden)
        return mean, log_std, carry


    def forward_sequence(self, obs_seq, done_seq, init_carry):
        """
        obs_seq:  [T,obs_dim]
        done_seq: [T] (done after step t)
        init_carry: [H] carry valid for obs_seq[0]
        """
        # reset BEFORE obs[t] if done at t-1 ; done_prev[0]=0
        done_prev = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), done_seq.astype(jnp.float32)[:-1]], axis=0)  # [T]

        def step(carry, inp):
            obs_t, done_prev_t = inp
            carry = carry * (1.0 - done_prev_t)
            mean_t, logstd_t, next_carry = self.apply_one_step(obs_t, carry)
            return next_carry, (mean_t, logstd_t)

        _, (mean_seq, logstd_seq) = jax.lax.scan(step, init_carry, (obs_seq, done_prev), unroll=True)

        return mean_seq, logstd_seq


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
