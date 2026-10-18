from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.trirl_dtrl.pytorch.trirl_dtrl import TRIRL_DTRL
from rl_x.algorithms.trirl_dtrl.pytorch.default_config import get_config
from rl_x.algorithms.trirl_dtrl.pytorch.general_properties import GeneralProperties


TRIRL_DTRL_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(TRIRL_DTRL_PYTORCH, get_config, TRIRL_DTRL, GeneralProperties)
