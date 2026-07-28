from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.td3.pytorch.td3 import TD3
from rl_x.algorithms.td3.pytorch.default_config import get_config
from rl_x.algorithms.td3.pytorch.general_properties import GeneralProperties


TD3_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(TD3_PYTORCH, get_config, TD3, GeneralProperties)
