from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.crossq.flax.crossq import CrossQ
from rl_x.algorithms.crossq.flax.default_config import get_config
from rl_x.algorithms.crossq.flax.general_properties import GeneralProperties


CROSSQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSQ_FLAX, get_config, CrossQ, GeneralProperties)
