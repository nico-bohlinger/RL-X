from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.redq.flax.redq import REDQ
from rl_x.algorithms.redq.flax.default_config import get_config
from rl_x.algorithms.redq.flax.general_properties import GeneralProperties


REDQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(REDQ_FLAX, get_config, REDQ, GeneralProperties)
