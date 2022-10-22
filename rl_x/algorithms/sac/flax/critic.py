import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.get_observation_space_type()

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(config.algorithm.nr_hidden_units, 2)
    else:
        raise ValueError(f"Unsupported observation_space_type: {observation_space_type}")


class Critic(nn.Module):
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x.squeeze()
    

class VectorCritic(nn.Module):
    nr_hidden_units: int
    nr_critics: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        # Reference:
        # - https://github.com/araffin/sbx/blob/f31288d2701b39dd98c921f55e13ca3530868e9f/sbx/sac/policies.py
        # - https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/networks/critic_net.py

        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )
        q_values = vmap_critic(nr_hidden_units=self.nr_hidden_units)(obs, action)
        return q_values
