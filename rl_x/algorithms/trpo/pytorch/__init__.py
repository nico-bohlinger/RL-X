from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trpo.pytorch.trpo import TRPO
from rl_x.algorithms.trpo.pytorch.default_config import get_config
from rl_x.algorithms.trpo.pytorch.general_properties import GeneralProperties


TRPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(TRPO_PYTORCH, get_config, TRPO, GeneralProperties)
