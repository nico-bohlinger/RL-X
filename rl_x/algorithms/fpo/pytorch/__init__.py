from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fpo.pytorch.fpo import FPO
from rl_x.algorithms.fpo.pytorch.default_config import get_config
from rl_x.algorithms.fpo.pytorch.general_properties import GeneralProperties


FPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(FPO_PYTORCH, get_config, FPO, GeneralProperties)
