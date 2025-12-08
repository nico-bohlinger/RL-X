from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fastsac.pytorch.fastsac import FastSAC
from rl_x.algorithms.fastsac.pytorch.default_config import get_config
from rl_x.algorithms.fastsac.pytorch.general_properties import GeneralProperties


FASTSAC_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTSAC_TORCHSCRIPT, get_config, FastSAC, GeneralProperties)