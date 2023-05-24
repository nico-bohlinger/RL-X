from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo.torchscript.ppo import PPO
from rl_x.algorithms.ppo.torchscript.default_config import get_config


PPO_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_TORCHSCRIPT, get_config, PPO)
