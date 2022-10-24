from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_droq.flax.tqc_droq import TQC_DroQ
from rl_x.algorithms.tqc_droq.flax.default_config import get_config


TQC_DROQ_FLAX = "TQC+DroQ Flax"
register_algorithm(TQC_DROQ_FLAX, get_config, TQC_DroQ)
