from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo.torchscript.ppo import PPO
from rl_x.algorithms.ppo.torchscript.default_config import get_config


PPO_TORCHSCRIPT = "PPO TorchScript"
register_algorithm(PPO_TORCHSCRIPT, get_config, PPO)
