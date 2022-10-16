from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.pytorch.sac import SAC
from rl_x.algorithms.sac.pytorch.default_config import get_config


SAC_PYTORCH = "SAC PyTorch"
register_algorithm(SAC_PYTORCH, get_config, SAC)
