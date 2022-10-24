from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc.flax.tqc import TQC
from rl_x.algorithms.tqc.flax.default_config import get_config


TQC_FLAX = "TQC Flax"
register_algorithm(TQC_FLAX, get_config, TQC)
