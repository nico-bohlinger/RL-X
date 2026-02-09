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
                    config.algorithm.window_hidden_dim,
                    config.algorithm.window_obs_combine_method,
                    config.algorithm.share_window_obs_encoder,
                    config.algorithm.window_length,
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
    window_hidden_dim: int
    window_obs_combine_method: str
    share_window_obs_encoder: bool
    window_length: int
    policy_observation_indices: Sequence[int]

    def setup(self):
        act_dim = int(np.prod(self.as_shape).item())

        self.window_obs_encoder_dense = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.window_obs_encoder_ln = nn.LayerNorm()

        if not self.share_window_obs_encoder:
            self.obs_encoder_dense = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.obs_encoder_ln = nn.LayerNorm()

        self.window_agg_dense = nn.Dense(self.window_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.window_agg_ln = nn.LayerNorm()

        if self.window_obs_combine_method == "film":
            self.window_film_gamma = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
            self.window_film_beta = nn.Dense(self.obs_encoding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.torso_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_ln1 = nn.LayerNorm()
        self.torso_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.torso_dense3 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))

        self.mean_head = nn.Dense(act_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.logstd = self.param("policy_logstd", constant(jnp.log(self.std_dev)), (1, np.prod(self.as_shape).item()))
    

    def initialize_window(self, nr_envs: int, obs_dim: int):
        return jnp.zeros((nr_envs, self.window_length, obs_dim), dtype=jnp.float32)


    def _reset_window(self, window):
        return jnp.zeros_like(window)


    def _update_window(self, window, obs):
        return jnp.concatenate([window[..., 1:, :], obs[..., None, :]], axis=-2)


    def obs_encode(self, obs):
        if self.share_window_obs_encoder:
            return self.window_obs_encode(obs)
        x = obs[..., self.policy_observation_indices]
        x = self.obs_encoder_dense(x)
        x = self.obs_encoder_ln(x)
        x = nn.elu(x)
        return x


    def window_obs_encode(self, obs):
        x = obs[..., self.policy_observation_indices]
        x = self.window_obs_encoder_dense(x)
        x = self.window_obs_encoder_ln(x)
        x = nn.elu(x)
        return x


    def window_encode(self, window):
        w_lat = self.window_obs_encode(window)  # (..., W, D)
        w_flat = w_lat.reshape(*w_lat.shape[:-2], -1)
        w = self.window_agg_dense(w_flat)
        w = self.window_agg_ln(w)
        w = nn.elu(w)
        return w


    def decode(self, obs_latent, window_latent):
        if self.window_obs_combine_method == "concat":
            torso_in = jnp.concatenate([obs_latent, window_latent], axis=-1)
        elif self.window_obs_combine_method == "film":
            gamma = self.window_film_gamma(window_latent)
            beta = self.window_film_beta(window_latent)
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
    

    def apply_one_step(self, obs, rolling_window):
        w_lat = self.window_encode(rolling_window)
        obs_lat = self.obs_encode(obs)
        mean, log_std = self.decode(obs_lat, w_lat)
        next_window = self._update_window(rolling_window, obs)
        return mean, log_std, next_window


    def forward_sequence(self, obs_seq, done_seq, init_window):
        """
        obs_seq:  [T,obs_dim]
        done_seq: [T] (done after step t)
        init_carry: [H] carry valid for obs_seq[0]
        """
        # reset BEFORE obs[t] if done at t-1 ; done_prev[0]=0
        done_prev = jnp.concatenate([jnp.zeros((1,), dtype=jnp.float32), done_seq.astype(jnp.float32)[:-1]], axis=0)  # [T]

        def step(window, inp):
            obs_t, done_prev_t = inp
            window = jnp.where(done_prev_t > 0.0, self._reset_window(window), window)
            mean_t, logstd_t, next_window = self.apply_one_step(obs_t, window)
            return next_window, (mean_t, logstd_t)
        
        _, (mean_seq, logstd_seq) = jax.lax.scan(step, init_window, (obs_seq, done_prev), unroll=True)

        return mean_seq, logstd_seq


def get_processed_action_function(action_clipping_and_rescaling, env_as_low, env_as_high):
    if action_clipping_and_rescaling:
        def get_clipped_and_scaled_action(action, env_as_low=env_as_low, env_as_high=env_as_high):
            clipped_action = jnp.clip(action, -1, 1)
            return env_as_low + (0.5 * (clipped_action + 1.0) * (env_as_high - env_as_low))
        return jax.jit(get_clipped_and_scaled_action)
    else:
        return jax.jit(lambda x: x)
