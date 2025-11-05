from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.pqn.flax.pqn import PQN
from rl_x.algorithms.pqn.flax.default_config import get_config
from rl_x.algorithms.pqn.flax.general_properties import GeneralProperties


PQN_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PQN_FLAX, get_config, PQN, GeneralProperties)
