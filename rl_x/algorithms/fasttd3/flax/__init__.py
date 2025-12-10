from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fasttd3.flax.fasttd3 import FastTD3
from rl_x.algorithms.fasttd3.flax.default_config import get_config
from rl_x.algorithms.fasttd3.flax.general_properties import GeneralProperties


FASTTD3_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTTD3_FLAX, get_config, FastTD3, GeneralProperties)