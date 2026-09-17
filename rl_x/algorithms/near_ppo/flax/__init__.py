from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.near_ppo.flax.near_ppo import NEAR_PPO
from rl_x.algorithms.near_ppo.flax.default_config import get_config
from rl_x.algorithms.near_ppo.flax.general_properties import GeneralProperties


NEAR_PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(NEAR_PPO_FLAX, get_config, NEAR_PPO, GeneralProperties)
