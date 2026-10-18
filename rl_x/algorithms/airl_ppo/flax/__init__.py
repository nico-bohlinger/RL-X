from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.airl_ppo.flax.airl_ppo import AIRL_PPO
from rl_x.algorithms.airl_ppo.flax.default_config import get_config
from rl_x.algorithms.airl_ppo.flax.general_properties import GeneralProperties


AIRL_PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(AIRL_PPO_FLAX, get_config, AIRL_PPO, GeneralProperties)
