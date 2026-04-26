import torch


class BoxSpace:
    """A Box space with center and scale attributes, based on robot locomotion environments. For other environments, these can be ignored.

    center: torch.Tensor
        In robot locomotion-like environments this is the nominal position of the joints. Has no impact on sampling.
    scale: torch.Tensor or float
        In robot locomotion-like environments this is a multiplier to scale the sampled actions. Has impact on sampling.
    """

    def __init__(self, low, high, shape, dtype, center=None, scale=None, device="cpu"):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.center = center if center is not None else torch.zeros(shape, dtype=dtype, device=device)
        self.scale = scale if scale is not None else torch.ones(shape, dtype=dtype, device=device)


    def sample(self, generator=None):
        low = self.low if torch.is_tensor(self.low) else torch.full(self.shape, float(self.low), dtype=self.dtype, device=self.device)
        high = self.high if torch.is_tensor(self.high) else torch.full(self.shape, float(self.high), dtype=self.dtype, device=self.device)
        sample = low + (high - low) * torch.rand(self.shape, dtype=self.dtype, device=self.device, generator=generator)
        scale = self.scale if torch.is_tensor(self.scale) else float(self.scale)
        return sample / scale
