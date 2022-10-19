from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.espo.pytorch.espo import ESPO
from rl_x.algorithms.espo.pytorch.default_config import get_config


ESPO_PYTORCH = "ESPO PyTorch"
register_algorithm(ESPO_PYTORCH, get_config, ESPO)
