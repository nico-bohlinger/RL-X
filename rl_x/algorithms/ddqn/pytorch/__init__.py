from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ddqn.pytorch.ddqn import DDQN
from rl_x.algorithms.ddqn.pytorch.default_config import get_config
from rl_x.algorithms.ddqn.pytorch.general_properties import GeneralProperties


DDQN_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DDQN_PYTORCH, get_config, DDQN, GeneralProperties)
