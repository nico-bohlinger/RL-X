from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_multihead.flax.tqc_multihead import TQC_MultiHead
from rl_x.algorithms.tqc_multihead.flax.default_config import get_config


TQC_MULTIHEAD_FLAX = "TQC MultiHead"
register_algorithm(TQC_MULTIHEAD_FLAX, get_config, TQC_MultiHead)
