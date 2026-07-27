from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.aqe.pytorch.aqe import AQE
from rl_x.algorithms.aqe.pytorch.default_config import get_config
from rl_x.algorithms.aqe.pytorch.general_properties import GeneralProperties


AQE_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(AQE_PYTORCH, get_config, AQE, GeneralProperties)
