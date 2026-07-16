from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.bro.flax.bro import BRO
from rl_x.algorithms.bro.flax.default_config import get_config
from rl_x.algorithms.bro.flax.general_properties import GeneralProperties


BRO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(BRO_FLAX, get_config, BRO, GeneralProperties)
