from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_gru.pytorch.ppo_gru import PPO_GRU
from rl_x.algorithms.ppo_gru.pytorch.default_config import get_config
from rl_x.algorithms.ppo_gru.pytorch.general_properties import GeneralProperties


PPO_GRU_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_GRU_PYTORCH, get_config, PPO_GRU, GeneralProperties)
