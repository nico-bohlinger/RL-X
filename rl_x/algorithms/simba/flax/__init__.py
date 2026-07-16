from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.simba.flax.simba import Simba
from rl_x.algorithms.simba.flax.default_config import get_config
from rl_x.algorithms.simba.flax.general_properties import GeneralProperties


SIMBA_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SIMBA_FLAX, get_config, Simba, GeneralProperties)
