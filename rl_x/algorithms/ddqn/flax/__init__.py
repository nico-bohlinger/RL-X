from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ddqn.flax.ddqn import DDQN
from rl_x.algorithms.ddqn.flax.default_config import get_config
from rl_x.algorithms.ddqn.flax.general_properties import GeneralProperties


DDQN_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DDQN_FLAX, get_config, DDQN, GeneralProperties)
