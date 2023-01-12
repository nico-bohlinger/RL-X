from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.mpo.flax.mpo import MPO
from rl_x.algorithms.mpo.flax.default_config import get_config


MPO_FLAX = "MPO Flax"
register_algorithm(MPO_FLAX, get_config, MPO)
