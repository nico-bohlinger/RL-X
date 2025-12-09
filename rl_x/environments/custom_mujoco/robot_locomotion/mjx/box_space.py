import jax


class BoxSpace:
    '''A Box space with center and scale attributes, based on robot locomotion environments. For other environments, these can be ignored.
    center: np.ndarray
        In robot locomotion-like environments this is the nominal position of the joints. Has no impact on sampling.
    scale: np.ndarray
        In robot locomotion-like environments this is a multiplier to scale the sampled actions. Has impact on sampling.
    '''

    def __init__(self, low, high, shape, dtype, center=None, scale=None):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.center = center if center is not None else jax.numpy.zeros(shape, dtype=dtype)
        self.scale = scale if scale is not None else jax.numpy.ones(shape, dtype=dtype)


    def sample(self, rng):
        return jax.random.uniform(rng, shape=self.shape, minval=self.low, maxval=self.high).astype(self.dtype)
