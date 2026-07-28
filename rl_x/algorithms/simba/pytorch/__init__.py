from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.simba.pytorch.simba import Simba
from rl_x.algorithms.simba.pytorch.default_config import get_config
from rl_x.algorithms.simba.pytorch.general_properties import GeneralProperties


SIMBA_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(SIMBA_PYTORCH, get_config, Simba, GeneralProperties)
