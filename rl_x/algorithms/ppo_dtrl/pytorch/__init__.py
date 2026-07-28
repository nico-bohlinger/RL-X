from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_dtrl.pytorch.ppo_dtrl import PPO_DTRL
from rl_x.algorithms.ppo_dtrl.pytorch.default_config import get_config
from rl_x.algorithms.ppo_dtrl.pytorch.general_properties import GeneralProperties


PPO_DTRL_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_DTRL_PYTORCH, get_config, PPO_DTRL, GeneralProperties)
