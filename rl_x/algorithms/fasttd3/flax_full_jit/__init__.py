from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fasttd3.flax_full_jit.fasttd3 import FastTD3
from rl_x.algorithms.fasttd3.flax_full_jit.default_config import get_config
from rl_x.algorithms.fasttd3.flax_full_jit.general_properties import GeneralProperties


FASTTD3_FULL_JIT = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTTD3_FULL_JIT, get_config, FastTD3, GeneralProperties)
