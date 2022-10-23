from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.redq.flax.redq import REDQ
from rl_x.algorithms.redq.flax.default_config import get_config


REDQ_FLAX = "REDQ Flax"
register_algorithm(REDQ_FLAX, get_config, REDQ)
