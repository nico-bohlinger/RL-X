from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.ppo.flax.ppo import PPO
from rl_x.algorithms.ppo.flax.default_config import get_config
from rl_x.algorithms.ppo.flax.general_properties import GeneralProperties


PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(PPO_FLAX, get_config, PPO, GeneralProperties)
