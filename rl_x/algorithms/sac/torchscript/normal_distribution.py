from typing import List
import math
import torch


# https://github.com/pytorch/pytorch/issues/29843#issuecomment-755290339
class Normal:
    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        resized_ = torch.broadcast_tensors(loc, scale)
        self.loc = resized_[0]
        self.scale = resized_[1]
        self._batch_shape = list(self.loc.size())

    def _extended_shape(self, sample_shape: List[int]) -> List[int]:
        return sample_shape + self._batch_shape

    def sample(self) -> torch.Tensor:
        shape = self._batch_shape
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))
    
    def rsample(self) -> torch.Tensor:
        shape = self._batch_shape
        eps = torch.normal(torch.zeros(shape, device=self.loc.device),
                           torch.ones(shape, device=self.scale.device))
        return self.loc + eps * self.scale

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        var = (self.scale ** 2)
        log_scale = self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def entropy(self) -> torch.Tensor:
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)
