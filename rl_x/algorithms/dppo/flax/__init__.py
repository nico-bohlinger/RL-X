from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dppo.flax.dppo import DPPO
from rl_x.algorithms.dppo.flax.default_config import get_config
from rl_x.algorithms.dppo.flax.general_properties import GeneralProperties


DPPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DPPO_FLAX, get_config, DPPO, GeneralProperties)
