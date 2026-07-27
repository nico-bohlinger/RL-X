from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dime.flax.dime import DIME
from rl_x.algorithms.dime.flax.default_config import get_config
from rl_x.algorithms.dime.flax.general_properties import GeneralProperties


DIME_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DIME_FLAX, get_config, DIME, GeneralProperties)
