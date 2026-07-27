from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.dime.pytorch.dime import DIME
from rl_x.algorithms.dime.pytorch.default_config import get_config
from rl_x.algorithms.dime.pytorch.general_properties import GeneralProperties


DIME_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(DIME_PYTORCH, get_config, DIME, GeneralProperties)
