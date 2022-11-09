from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_softwatkins.flax.tqc_softwatkins import TQC_SoftWatkinsQLambda
from rl_x.algorithms.tqc_softwatkins.flax.default_config import get_config


TQC_SOFTWATKINS_FLAX = "TQC+SoftWatkinsQLambda Flax"
register_algorithm(TQC_SOFTWATKINS_FLAX, get_config, TQC_SoftWatkinsQLambda)
