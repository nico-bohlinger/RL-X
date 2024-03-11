from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dqn_hl_gauss.flax.dqn_hl_gauss import DQN_HL_Gauss
from rl_x.algorithms.dqn_hl_gauss.flax.default_config import get_config
from rl_x.algorithms.dqn_hl_gauss.flax.general_properties import GeneralProperties


DQN_HL_GAUSS_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(DQN_HL_GAUSS_FLAX, get_config, DQN_HL_Gauss, GeneralProperties)
