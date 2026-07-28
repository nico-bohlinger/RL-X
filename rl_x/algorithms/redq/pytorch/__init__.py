from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.redq.pytorch.redq import REDQ
from rl_x.algorithms.redq.pytorch.default_config import get_config
from rl_x.algorithms.redq.pytorch.general_properties import GeneralProperties


REDQ_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(REDQ_PYTORCH, get_config, REDQ, GeneralProperties)
