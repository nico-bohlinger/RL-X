from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.aqe.flax.aqe import AQE
from rl_x.algorithms.aqe.flax.default_config import get_config


AQE_FLAX = "AQE Flax"
register_algorithm(AQE_FLAX, get_config, AQE)
