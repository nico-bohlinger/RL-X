from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.droq.flax.droq import DroQ
from rl_x.algorithms.droq.flax.default_config import get_config


DROQ_FLAX = "DroQ Flax"
register_algorithm(DROQ_FLAX, get_config, DroQ)
