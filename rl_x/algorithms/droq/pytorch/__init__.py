from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.droq.pytorch.droq import DroQ
from rl_x.algorithms.droq.pytorch.default_config import get_config
from rl_x.algorithms.droq.pytorch.general_properties import GeneralProperties


DROQ_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DROQ_PYTORCH, get_config, DroQ, GeneralProperties)
