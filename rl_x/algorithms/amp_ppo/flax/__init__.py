from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.amp_ppo.flax.amp_ppo import AMP_PPO
from rl_x.algorithms.amp_ppo.flax.default_config import get_config
from rl_x.algorithms.amp_ppo.flax.general_properties import GeneralProperties


AMP_PPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(AMP_PPO_FLAX, get_config, AMP_PPO, GeneralProperties)
