from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trirl_dtrl.flax.trirl_dtrl import TRIRL_DTRL
from rl_x.algorithms.trirl_dtrl.flax.default_config import get_config
from rl_x.algorithms.trirl_dtrl.flax.general_properties import GeneralProperties


TRIRL_DTRL_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TRIRL_DTRL_FLAX, get_config, TRIRL_DTRL, GeneralProperties)
