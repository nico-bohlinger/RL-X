from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dppo.flax_full_jit.dppo import DPPO
from rl_x.algorithms.dppo.flax_full_jit.default_config import get_config
from rl_x.algorithms.dppo.flax_full_jit.general_properties import GeneralProperties


DPPO_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(DPPO_FLAX_FULL_JIT, get_config, DPPO, GeneralProperties)
