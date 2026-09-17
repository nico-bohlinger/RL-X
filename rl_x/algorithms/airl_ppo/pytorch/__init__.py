from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.airl_ppo.pytorch.airl_ppo import AIRL_PPO
from rl_x.algorithms.airl_ppo.pytorch.default_config import get_config
from rl_x.algorithms.airl_ppo.pytorch.general_properties import GeneralProperties


AIRL_PPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(AIRL_PPO_PYTORCH, get_config, AIRL_PPO, GeneralProperties)
