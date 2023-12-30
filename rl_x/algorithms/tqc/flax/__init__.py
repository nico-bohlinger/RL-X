from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.tqc.flax.tqc import TQC
from rl_x.algorithms.tqc.flax.default_config import get_config
from rl_x.algorithms.tqc.flax.general_properties import GeneralProperties


TQC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TQC_FLAX, get_config, TQC, GeneralProperties)
