from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_mamba2.flax.ppo_mamba2 import PPO_Mamba2
from rl_x.algorithms.ppo_mamba2.flax.default_config import get_config
from rl_x.algorithms.ppo_mamba2.flax.general_properties import GeneralProperties


PPO_MAMBA2_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_MAMBA2_FLAX, get_config, PPO_Mamba2, GeneralProperties)
