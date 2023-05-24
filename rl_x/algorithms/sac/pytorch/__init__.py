from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac.pytorch.sac import SAC
from rl_x.algorithms.sac.pytorch.default_config import get_config


SAC_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_PYTORCH, get_config, SAC)
