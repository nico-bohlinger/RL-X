from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_tanhinitfix.flax.tqc_tanhinitfix import TQC_TanhInitFix
from rl_x.algorithms.tqc_tanhinitfix.flax.default_config import get_config


TQC_TANHINITFIX_FLAX = "TQC+TanhInitFix Flax"
register_algorithm(TQC_TANHINITFIX_FLAX, get_config, TQC_TanhInitFix)
