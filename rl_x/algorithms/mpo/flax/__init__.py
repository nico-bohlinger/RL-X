from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.mpo.flax.mpo import MPO
from rl_x.algorithms.mpo.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

MPO_FLAX = algorithm_name
register_algorithm(MPO_FLAX, get_config, MPO)
