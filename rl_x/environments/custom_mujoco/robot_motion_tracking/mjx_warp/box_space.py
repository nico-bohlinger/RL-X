import jax


class BoxSpace:
    """A Box space with an optional action center and scale."""

    def __init__(self, low, high, shape, dtype, center=None, scale=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.center = center if center is not None else jax.numpy.zeros(shape, dtype=dtype)
        self.scale = scale if scale is not None else jax.numpy.ones(shape, dtype=dtype)


    def sample(self, rng):
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype) / self.scale
