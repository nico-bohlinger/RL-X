from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.spo.flax.spo import SPO
from rl_x.algorithms.spo.flax.default_config import get_config
from rl_x.algorithms.spo.flax.general_properties import GeneralProperties


SPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(SPO_FLAX, get_config, SPO, GeneralProperties)
