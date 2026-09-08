import gymnasium as gym
import numpy as np


class BoxSpace(gym.spaces.Box):
    '''A Box space with center and scale attributes, based on robot locomotion environments. For other environments, these can be ignored.
    center: np.ndarray
        In robot locomotion-like environments this is the nominal position of the joints. Has no impact on sampling.
    scale: np.ndarray
        In robot locomotion-like environments this is a multiplier to scale the sampled actions. Has impact on sampling.
    '''

    def __init__(self, low, high, shape, dtype, center=None, scale=None):
        super().__init__(low=low, high=high, shape=shape, dtype=dtype)
        self.center = center if center is not None else np.zeros(shape, dtype=dtype)
        self.scale = scale if scale is not None else np.ones(shape, dtype=dtype)
        
        self.center = gym.spaces.box._broadcast(self.center, self.dtype, shape)
        self.center = self.center.astype(self.dtype)
        
        self.scale = gym.spaces.box._broadcast(self.scale, self.dtype, shape)
        self.scale = self.scale.astype(self.dtype)


    def sample(self):
        sample = super().sample()
        sample = sample / self.scale
        return sample
