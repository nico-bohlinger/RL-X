from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trpo.flax.trpo import TRPO
from rl_x.algorithms.trpo.flax.default_config import get_config
from rl_x.algorithms.trpo.flax.general_properties import GeneralProperties


TRPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TRPO_FLAX, get_config, TRPO, GeneralProperties)
