from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ddpg.flax.ddpg import DDPG
from rl_x.algorithms.ddpg.flax.default_config import get_config
from rl_x.algorithms.ddpg.flax.general_properties import GeneralProperties


DDPG_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DDPG_FLAX, get_config, DDPG, GeneralProperties)
