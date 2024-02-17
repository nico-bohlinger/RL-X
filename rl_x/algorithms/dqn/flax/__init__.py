from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dqn.flax.dqn import DQN
from rl_x.algorithms.dqn.flax.default_config import get_config
from rl_x.algorithms.dqn.flax.general_properties import GeneralProperties


DQN_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DQN_FLAX, get_config, DQN, GeneralProperties)
