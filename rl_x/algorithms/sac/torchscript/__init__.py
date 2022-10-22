from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.torchscript.sac import SAC
from rl_x.algorithms.sac.torchscript.default_config import get_config


SAC_TORCHSCRIPT = "SAC TorchScript"
register_algorithm(SAC_TORCHSCRIPT, get_config, SAC)
