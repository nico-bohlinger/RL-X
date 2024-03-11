import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.IMAGES:
        return Critic(env.get_single_action_logit_size(), config.algorithm.nr_hidden_units)


class Critic(nn.Module):
    nr_available_actions: int
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: np.ndarray):
        x = x / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.nr_available_actions)(x)
        return x
