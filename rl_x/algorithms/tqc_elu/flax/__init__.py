from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_elu.flax.tqc_elu import TQC_ELU
from rl_x.algorithms.tqc_elu.flax.default_config import get_config


TQC_ELU_FLAX = "TQC+ELU Flax"
register_algorithm(TQC_ELU_FLAX, get_config, TQC_ELU)
