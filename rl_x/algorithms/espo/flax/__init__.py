from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.espo.flax.espo import ESPO
from rl_x.algorithms.espo.flax.default_config import get_config
from rl_x.algorithms.espo.flax.general_properties import GeneralProperties


ESPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(ESPO_FLAX, get_config, ESPO, GeneralProperties)
