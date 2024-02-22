from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.td3.flax.td3 import TD3
from rl_x.algorithms.td3.flax.default_config import get_config
from rl_x.algorithms.td3.flax.general_properties import GeneralProperties


TD3_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(TD3_FLAX, get_config, TD3, GeneralProperties)
