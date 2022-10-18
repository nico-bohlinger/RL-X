from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.flax.ppo import PPO
from rl_x.algorithms.ppo.flax.default_config import get_config


PPO_FLAX = "PPO Flax"
register_algorithm(PPO_FLAX, get_config, PPO)
