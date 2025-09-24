from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trpl.flax.trpl import TRPL
from rl_x.algorithms.trpl.flax.default_config import get_config
from rl_x.algorithms.trpl.flax.general_properties import GeneralProperties

TRPL_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TRPL_FLAX, get_config, TRPL, GeneralProperties)
