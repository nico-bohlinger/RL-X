from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.flax.sac import SAC
from rl_x.algorithms.sac.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

SAC_FLAX = algorithm_name
register_algorithm(SAC_FLAX, get_config, SAC)
