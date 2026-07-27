from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.pqn.pytorch.pqn import PQN
from rl_x.algorithms.pqn.pytorch.default_config import get_config
from rl_x.algorithms.pqn.pytorch.general_properties import GeneralProperties


PQN_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(PQN_PYTORCH, get_config, PQN, GeneralProperties)
