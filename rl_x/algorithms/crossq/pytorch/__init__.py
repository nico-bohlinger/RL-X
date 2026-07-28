from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.crossq.pytorch.crossq import CrossQ
from rl_x.algorithms.crossq.pytorch.default_config import get_config
from rl_x.algorithms.crossq.pytorch.general_properties import GeneralProperties


CROSSQ_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(CROSSQ_PYTORCH, get_config, CrossQ, GeneralProperties)
