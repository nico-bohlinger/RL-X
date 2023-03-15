from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.espo.flax.espo import ESPO
from rl_x.algorithms.espo.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

ESPO_FLAX = algorithm_name
register_algorithm(ESPO_FLAX, get_config, ESPO)
