from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.c51.pytorch.c51 import C51
from rl_x.algorithms.c51.pytorch.default_config import get_config
from rl_x.algorithms.c51.pytorch.general_properties import GeneralProperties


C51_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(C51_PYTORCH, get_config, C51, GeneralProperties)
