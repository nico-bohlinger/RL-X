from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_retrace.flax.tqc_retrace import TQC_Retrace
from rl_x.algorithms.tqc_retrace.flax.default_config import get_config


TQC_RETRACE_FLAX = "TQC+Retrace Flax"
register_algorithm(TQC_RETRACE_FLAX, get_config, TQC_Retrace)
