from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.espo.pytorch.espo import ESPO
from rl_x.algorithms.espo.pytorch.default_config import get_config
from rl_x.algorithms.espo.pytorch.general_properties import GeneralProperties


ESPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(ESPO_PYTORCH, get_config, ESPO, GeneralProperties)
