from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.gail_ppo.flax.gail_ppo import GAIL_PPO
from rl_x.algorithms.gail_ppo.flax.default_config import get_config
from rl_x.algorithms.gail_ppo.flax.general_properties import GeneralProperties


GAIL_PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(GAIL_PPO_FLAX, get_config, GAIL_PPO, GeneralProperties)
