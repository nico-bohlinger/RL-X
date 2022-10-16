from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.flax.sac import SAC
from rl_x.algorithms.sac.flax.default_config import get_config


SAC_FLAX = "SAC Flax"
register_algorithm(SAC_FLAX, get_config, SAC)
