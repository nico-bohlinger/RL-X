from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.droq.flax.droq import DroQ
from rl_x.algorithms.droq.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

DROQ_FLAX = algorithm_name
register_algorithm(DROQ_FLAX, get_config, DroQ)
