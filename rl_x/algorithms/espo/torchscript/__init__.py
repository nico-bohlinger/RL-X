from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.espo.torchscript.espo import ESPO
from rl_x.algorithms.espo.torchscript.default_config import get_config


ESPO_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(ESPO_TORCHSCRIPT, get_config, ESPO)
