from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dime.flax_full_jit.dime import DIME
from rl_x.algorithms.dime.flax_full_jit.default_config import get_config
from rl_x.algorithms.dime.flax_full_jit.general_properties import GeneralProperties


DIME_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(DIME_FLAX_FULL_JIT, get_config, DIME, GeneralProperties)
