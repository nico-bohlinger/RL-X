from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.reppo.pytorch.reppo import REPPO
from rl_x.algorithms.reppo.pytorch.default_config import get_config
from rl_x.algorithms.reppo.pytorch.general_properties import GeneralProperties


REPPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(REPPO_PYTORCH, get_config, REPPO, GeneralProperties)
