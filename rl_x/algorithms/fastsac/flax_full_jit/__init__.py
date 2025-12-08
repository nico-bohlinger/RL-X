from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fastsac.flax_full_jit.fastsac import FastSAC
from rl_x.algorithms.fastsac.flax_full_jit.default_config import get_config
from rl_x.algorithms.fastsac.flax_full_jit.general_properties import GeneralProperties


FASTSAC_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTSAC_FULL_JIT, get_config, FastSAC, GeneralProperties)