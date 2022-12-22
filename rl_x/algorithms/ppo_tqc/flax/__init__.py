from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.ppo_tqc.flax.ppo_tqc import PPO_TQC
from rl_x.algorithms.ppo_tqc.flax.default_config import get_config


PPO_TQC_FLAX = "PPO+TQC Flax"
register_algorithm(PPO_TQC_FLAX, get_config, PPO_TQC)
