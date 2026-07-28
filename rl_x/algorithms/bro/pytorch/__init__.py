from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.bro.pytorch.bro import BRO
from rl_x.algorithms.bro.pytorch.default_config import get_config
from rl_x.algorithms.bro.pytorch.general_properties import GeneralProperties


BRO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(BRO_PYTORCH, get_config, BRO, GeneralProperties)
