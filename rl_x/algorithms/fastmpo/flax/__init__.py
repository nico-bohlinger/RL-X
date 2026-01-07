from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fastmpo.flax.fastmpo import FastMPO
from rl_x.algorithms.fastmpo.flax.default_config import get_config
from rl_x.algorithms.fastmpo.flax.general_properties import GeneralProperties


FASTMPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTMPO_FLAX, get_config, FastMPO, GeneralProperties)