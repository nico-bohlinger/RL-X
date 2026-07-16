import jax.numpy as jnp


class NoneActionDelay:
    def __init__(self, env):
        self.env = env


    def init(self, internal_state):
        pass


    def setup(self, internal_state):
        pass


    def sample(self, internal_state, should_randomize, key):
        pass


    def delay_action(self, action, internal_state):
        return jnp.broadcast_to(action, (self.env.nr_substeps,) + action.shape)
