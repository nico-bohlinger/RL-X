from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.aqe.flax.aqe import AQE
from rl_x.algorithms.aqe.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

AQE_FLAX = algorithm_name
register_algorithm(AQE_FLAX, get_config, AQE)
