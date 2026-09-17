from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.amp_ppo.pytorch.amp_ppo import AMP_PPO
from rl_x.algorithms.amp_ppo.pytorch.default_config import get_config
from rl_x.algorithms.amp_ppo.pytorch.general_properties import GeneralProperties


AMP_PPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(AMP_PPO_PYTORCH, get_config, AMP_PPO, GeneralProperties)
