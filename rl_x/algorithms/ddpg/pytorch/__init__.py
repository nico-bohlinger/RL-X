from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ddpg.pytorch.ddpg import DDPG
from rl_x.algorithms.ddpg.pytorch.default_config import get_config
from rl_x.algorithms.ddpg.pytorch.general_properties import GeneralProperties


DDPG_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DDPG_PYTORCH, get_config, DDPG, GeneralProperties)
