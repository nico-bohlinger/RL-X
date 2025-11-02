from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo_dtrl.flax.ppo_dtrl import PPO_DTRL
from rl_x.algorithms.ppo_dtrl.flax.default_config import get_config
from rl_x.algorithms.ppo_dtrl.flax.general_properties import GeneralProperties

PPO_DTRL_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_DTRL_FLAX, get_config, PPO_DTRL, GeneralProperties)
