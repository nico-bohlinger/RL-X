from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trpo.flax_full_jit.trpo import TRPO
from rl_x.algorithms.trpo.flax_full_jit.default_config import get_config
from rl_x.algorithms.trpo.flax_full_jit.general_properties import GeneralProperties


TRPO_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(TRPO_FLAX_FULL_JIT, get_config, TRPO, GeneralProperties)
