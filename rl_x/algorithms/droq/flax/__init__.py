from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.droq.flax.droq import DroQ
from rl_x.algorithms.droq.flax.default_config import get_config
from rl_x.algorithms.droq.flax.general_properties import GeneralProperties


DROQ_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DROQ_FLAX, get_config, DroQ, GeneralProperties)
