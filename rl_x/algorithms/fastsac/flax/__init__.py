from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fastsac.flax.fastsac import FastSAC
from rl_x.algorithms.fastsac.flax.default_config import get_config
from rl_x.algorithms.fastsac.flax.general_properties import GeneralProperties


FASTSAC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTSAC_FLAX, get_config, FastSAC, GeneralProperties)