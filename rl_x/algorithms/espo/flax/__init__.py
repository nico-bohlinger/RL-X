from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.espo.flax.espo import ESPO
from rl_x.algorithms.espo.flax.default_config import get_config


ESPO_FLAX = "ESPO Flax"
register_algorithm(ESPO_FLAX, get_config, ESPO)
