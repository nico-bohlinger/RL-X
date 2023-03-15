from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.redq.flax.redq import REDQ
from rl_x.algorithms.redq.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

REDQ_FLAX = algorithm_name
register_algorithm(REDQ_FLAX, get_config, REDQ)
