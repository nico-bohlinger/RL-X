from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.flax.ppo import PPO
from rl_x.algorithms.ppo.flax.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

PPO_FLAX = algorithm_name
register_algorithm(PPO_FLAX, get_config, PPO)
