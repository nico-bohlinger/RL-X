from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac.flax.sac import SAC
from rl_x.algorithms.sac.flax.default_config import get_config
from rl_x.algorithms.sac.flax.general_properties import GeneralProperties


SAC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_FLAX, get_config, SAC, GeneralProperties)
