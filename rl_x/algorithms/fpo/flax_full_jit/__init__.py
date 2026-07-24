from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fpo.flax_full_jit.fpo import FPO
from rl_x.algorithms.fpo.flax_full_jit.default_config import get_config
from rl_x.algorithms.fpo.flax_full_jit.general_properties import GeneralProperties


FPO_FLAX_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(FPO_FLAX_FULL_JIT, get_config, FPO, GeneralProperties)
