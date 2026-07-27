from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.simbav2.pytorch.simbav2 import SimbaV2
from rl_x.algorithms.simbav2.pytorch.default_config import get_config
from rl_x.algorithms.simbav2.pytorch.general_properties import GeneralProperties


SIMBAV2_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(SIMBAV2_PYTORCH, get_config, SimbaV2, GeneralProperties)
