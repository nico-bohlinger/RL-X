from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trirl_ppo.pytorch.trirl_ppo import TRIRL_PPO
from rl_x.algorithms.trirl_ppo.pytorch.default_config import get_config
from rl_x.algorithms.trirl_ppo.pytorch.general_properties import GeneralProperties


TRIRL_PPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(TRIRL_PPO_PYTORCH, get_config, TRIRL_PPO, GeneralProperties)
