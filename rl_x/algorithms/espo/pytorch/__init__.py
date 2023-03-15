from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.espo.pytorch.espo import ESPO
from rl_x.algorithms.espo.pytorch.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

ESPO_PYTORCH = algorithm_name
register_algorithm(ESPO_PYTORCH, get_config, ESPO)
