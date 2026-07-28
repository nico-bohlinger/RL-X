from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dppo.pytorch.dppo import DPPO
from rl_x.algorithms.dppo.pytorch.default_config import get_config
from rl_x.algorithms.dppo.pytorch.general_properties import GeneralProperties


DPPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DPPO_PYTORCH, get_config, DPPO, GeneralProperties)
