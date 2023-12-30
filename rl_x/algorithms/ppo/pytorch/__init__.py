from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo.pytorch.ppo import PPO
from rl_x.algorithms.ppo.pytorch.default_config import get_config
from rl_x.algorithms.ppo.pytorch.general_properties import GeneralProperties


PPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_PYTORCH, get_config, PPO, GeneralProperties)
