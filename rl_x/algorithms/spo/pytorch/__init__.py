from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.spo.pytorch.spo import SPO
from rl_x.algorithms.spo.pytorch.default_config import get_config
from rl_x.algorithms.spo.pytorch.general_properties import GeneralProperties


SPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(SPO_PYTORCH, get_config, SPO, GeneralProperties)
