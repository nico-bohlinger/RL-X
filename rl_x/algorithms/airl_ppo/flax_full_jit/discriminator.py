import numpy as np
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from collections import deque

def get_discriminator(config, env):
    action_space_type = env.general_properties.action_space_type
    observation_space_type = env.general_properties.observation_space_type

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return DiscriminatorShaped(config.algorithm.handle_absorbing_states, config.algorithm.gamma)


class DiscriminatorShaped(nn.Module):
    handle_absorbing_states: bool
    gamma: int

    def setup(self):
        self.gnet_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.gnet_dense3 = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))

        self.hnet_dense1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense2 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.hnet_dense3 = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))

    def __call__(self, x, a, x_n, absorbing, logp, shaping: float = 1.0):
        """
        D(s) + gamma h(s') - h(s)
        Args:
            x : state
            a: action (not used)
            x_n: next state
            abs: bool for whether the next state is absorbing
        """
        # g(x)
        r = self.gnet_dense1(x)
        r = nn.tanh(r)
        r = self.gnet_dense2(r)
        r = nn.tanh(r)
        r = self.gnet_dense3(r)

        # g(x_n)
        rx_n = self.gnet_dense1(x_n)
        rx_n = nn.tanh(rx_n)
        rx_n = self.gnet_dense2(rx_n)
        rx_n = nn.tanh(rx_n)
        rx_n = self.gnet_dense3(rx_n)

        # h(x)
        hx = self.hnet_dense1(x)
        hx = nn.tanh(hx)
        hx = self.hnet_dense2(hx)
        hx = nn.tanh(hx)
        hx = self.hnet_dense3(hx)

        # h(x_n)
        hx_n = self.hnet_dense1(x_n)
        hx_n = nn.tanh(hx_n)
        hx_n = self.hnet_dense2(hx_n)
        hx_n = nn.tanh(hx_n)
        hx_n = self.hnet_dense3(hx_n)

        absorbing = jnp.asarray(absorbing, dtype=r.dtype)
        absorbing = jnp.broadcast_to(absorbing[..., None], r.shape)

        # Shaped reward: step. 6 in alg 1 in https://arxiv.org/pdf/1710.11248v2
        if self.handle_absorbing_states:
            f = r + shaping * ((1 - absorbing) * self.gamma * hx_n + absorbing * ((self.gamma/(1 - self.gamma)) * rx_n) - hx)
        else:
            f = r + shaping * (self.gamma * hx_n - hx)
        reward = f - shaping * logp # D = sigmoid(reward)

        return reward.squeeze(-1)