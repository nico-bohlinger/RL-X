from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.reppo.flax.reppo import REPPO
from rl_x.algorithms.reppo.flax.default_config import get_config
from rl_x.algorithms.reppo.flax.general_properties import GeneralProperties


REPPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(REPPO_FLAX, get_config, REPPO, GeneralProperties)
