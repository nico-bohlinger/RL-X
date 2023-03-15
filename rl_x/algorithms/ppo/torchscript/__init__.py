from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.torchscript.ppo import PPO
from rl_x.algorithms.ppo.torchscript.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

PPO_TORCHSCRIPT = algorithm_name
register_algorithm(PPO_TORCHSCRIPT, get_config, PPO)
