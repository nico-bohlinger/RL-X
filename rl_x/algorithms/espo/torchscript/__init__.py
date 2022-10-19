from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.espo.torchscript.espo import ESPO
from rl_x.algorithms.espo.torchscript.default_config import get_config


ESPO_TORCHSCRIPT = "ESPO TorchScript"
register_algorithm(ESPO_TORCHSCRIPT, get_config, ESPO)
