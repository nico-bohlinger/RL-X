from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fasttd3.pytorch.fasttd3 import FastTD3
from rl_x.algorithms.fasttd3.pytorch.default_config import get_config
from rl_x.algorithms.fasttd3.pytorch.general_properties import GeneralProperties


FASTTD3_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTTD3_TORCHSCRIPT, get_config, FastTD3, GeneralProperties)