from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_per.flax.tqc_per import TQC_PER
from rl_x.algorithms.tqc_per.flax.default_config import get_config


TQC_PER_FLAX = "TQC+PER Flax"
register_algorithm(TQC_PER_FLAX, get_config, TQC_PER)
