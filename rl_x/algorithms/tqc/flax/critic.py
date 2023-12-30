import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(config.algorithm.nr_atoms_per_net, config.algorithm.nr_hidden_units, config.algorithm.ensemble_size)


class Critic(nn.Module):
    nr_atoms: int
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray):
        x = jnp.concatenate([x, a], -1)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_atoms)(x)
        return x
    

class VectorCritic(nn.Module):
    nr_atoms_per_net: int
    nr_hidden_units: int
    nr_critics: int

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray):
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
        q_values = vmap_critic(nr_atoms=self.nr_atoms_per_net, nr_hidden_units=self.nr_hidden_units)(obs, action)
        return q_values
