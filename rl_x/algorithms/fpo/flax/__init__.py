from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fpo.flax.fpo import FPO
from rl_x.algorithms.fpo.flax.default_config import get_config
from rl_x.algorithms.fpo.flax.general_properties import GeneralProperties


FPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(FPO_FLAX, get_config, FPO, GeneralProperties)
