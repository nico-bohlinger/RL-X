import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from rl_x.algorithms.crossq.flax.batch_renorm import BatchRenorm

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return VectorCritic(
            config.algorithm.batch_renorm_momentum,
            config.algorithm.batch_renorm_warmup_steps,
            config.algorithm.critic_nr_hidden_units,
            config.algorithm.ensemble_size
        )


class Critic(nn.Module):
    batch_renorm_momentum: float
    batch_renorm_warmup_steps: int
    nr_hidden_units: int

    @nn.compact
    def __call__(self, x: np.ndarray, a: np.ndarray, train):
        x = jnp.concatenate([x, a], -1)

        x = BatchRenorm(
            use_running_average=not train,
            momentum=self.batch_renorm_momentum,
            warm_up_steps=self.batch_renorm_warmup_steps
        )(x)

        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = BatchRenorm(
            use_running_average=not train,
            momentum=self.batch_renorm_momentum,
            warm_up_steps=self.batch_renorm_warmup_steps
        )(x)

        x = nn.Dense(self.nr_hidden_units)(x)
        x = nn.relu(x)
        x = BatchRenorm(
            use_running_average=not train,
            momentum=self.batch_renorm_momentum,
            warm_up_steps=self.batch_renorm_warmup_steps
        )(x)

        x = nn.Dense(1)(x)

        return x
    

class VectorCritic(nn.Module):
    batch_renorm_momentum: float
    batch_renorm_warmup_steps: int
    nr_hidden_units: int
    nr_critics: int

    @nn.compact
    def __call__(self, obs: np.ndarray, action: np.ndarray, train):
        # Reference:
        # - https://github.com/araffin/sbx/blob/f31288d2701b39dd98c921f55e13ca3530868e9f/sbx/sac/policies.py
        # - https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/networks/critic_net.py

        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0, "batch_stats": 0},  # parameters and batch stats not shared between the critics
            split_rngs={"params": True, "batch_stats": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.nr_critics,
        )

        q_values = vmap_critic(
            batch_renorm_momentum=self.batch_renorm_momentum,
            batch_renorm_warmup_steps=self.batch_renorm_warmup_steps,
            nr_hidden_units=self.nr_hidden_units
        )(obs, action, train)

        return q_values
