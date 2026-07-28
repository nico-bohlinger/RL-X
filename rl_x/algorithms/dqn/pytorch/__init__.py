from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dqn.pytorch.dqn import DQN
from rl_x.algorithms.dqn.pytorch.default_config import get_config
from rl_x.algorithms.dqn.pytorch.general_properties import GeneralProperties


DQN_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DQN_PYTORCH, get_config, DQN, GeneralProperties)
