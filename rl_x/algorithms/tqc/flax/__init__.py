from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc.flax.tqc import TQC
from rl_x.algorithms.tqc.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

TQC_FLAX = algorithm_name
register_algorithm(TQC_FLAX, get_config, TQC)
