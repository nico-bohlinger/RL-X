from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.mpo.flax.mpo import MPO
from rl_x.algorithms.mpo.flax.default_config import get_config
from rl_x.algorithms.mpo.flax.general_properties import GeneralProperties


MPO_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(MPO_FLAX, get_config, MPO, GeneralProperties)
