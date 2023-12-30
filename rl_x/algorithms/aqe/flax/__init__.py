from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.aqe.flax.aqe import AQE
from rl_x.algorithms.aqe.flax.default_config import get_config
from rl_x.algorithms.aqe.flax.general_properties import GeneralProperties


AQE_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(AQE_FLAX, get_config, AQE, GeneralProperties)
