from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac.torchscript.sac import SAC
from rl_x.algorithms.sac.torchscript.default_config import get_config


SAC_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_TORCHSCRIPT, get_config, SAC)
