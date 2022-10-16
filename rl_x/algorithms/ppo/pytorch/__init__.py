from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.pytorch.ppo import PPO
from rl_x.algorithms.ppo.pytorch.default_config import get_config


PPO_PYTORCH = "PPO PyTorch"
register_algorithm(PPO_PYTORCH, get_config, PPO)
