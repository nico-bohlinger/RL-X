from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.mpo.pytorch.mpo import MPO
from rl_x.algorithms.mpo.pytorch.default_config import get_config
from rl_x.algorithms.mpo.pytorch.general_properties import GeneralProperties


MPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(MPO_PYTORCH, get_config, MPO, GeneralProperties)