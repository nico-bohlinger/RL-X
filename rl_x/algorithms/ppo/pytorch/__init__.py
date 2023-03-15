from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.pytorch.ppo import PPO
from rl_x.algorithms.ppo.pytorch.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

PPO_PYTORCH = algorithm_name
register_algorithm(PPO_PYTORCH, get_config, PPO)
