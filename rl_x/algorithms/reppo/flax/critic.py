from typing import Sequence
import jax.numpy as jnp
from jax.scipy.special import erf
import flax.linen as nn
from flax.linen.initializers import constant

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type
    critic_observation_indices = getattr(env, "critic_observation_indices", jnp.arange(env.single_observation_space.shape[0]))

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(
            hidden_dim=config.algorithm.critic_hidden_dim,
            nr_bins=config.algorithm.nr_bins,
            v_min=config.algorithm.v_min,
            v_max=config.algorithm.v_max,
            critic_observation_indices=critic_observation_indices,
        )


class Critic(nn.Module):
    hidden_dim: int
    nr_bins: int
    v_min: float
    v_max: float
    critic_observation_indices: Sequence[int]

    @nn.compact
    def __call__(self, obs, action):
        obs = obs[..., self.critic_observation_indices]
        x = jnp.concatenate([obs, action], axis=-1)

        features = nn.Dense(self.hidden_dim)(x)
        features = nn.RMSNorm()(features)
        features = nn.swish(features)
        features = nn.Dense(self.hidden_dim)(features)

        critic_h = nn.Dense(self.hidden_dim)(nn.swish(features))
        critic_h = nn.RMSNorm()(critic_h)
        critic_h = nn.swish(critic_h)
        critic_logits = nn.Dense(self.nr_bins)(critic_h)

        bin_width = (self.v_max - self.v_min) / (self.nr_bins - 1)
        support = jnp.linspace(self.v_min - bin_width / 2, self.v_max + bin_width / 2, self.nr_bins + 1, dtype=jnp.float32)
        cdf_evals = erf(support / (jnp.sqrt(2) * bin_width * 0.75))
        zero_distribution = (cdf_evals[1:] - cdf_evals[:-1]) / (cdf_evals[-1] - cdf_evals[0])
        zero_distribution = self.param("zero_distribution", constant(zero_distribution), (self.nr_bins,))
        critic_logits = critic_logits + 40.0 * zero_distribution

        pred_h = nn.Dense(self.hidden_dim)(nn.swish(features))
        pred_h = nn.RMSNorm()(pred_h)
        pred_h = nn.swish(pred_h)
        pred = nn.Dense(self.hidden_dim + 1)(pred_h)
        pred_features = pred[..., 1:]
        pred_reward = pred[..., :1]

        return features, critic_logits, pred_features, pred_reward
