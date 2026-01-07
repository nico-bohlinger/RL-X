from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.fastmpo.pytorch.fastmpo import FastMPO
from rl_x.algorithms.fastmpo.pytorch.default_config import get_config
from rl_x.algorithms.fastmpo.pytorch.general_properties import GeneralProperties


FASTMPO_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(FASTMPO_PYTORCH, get_config, FastMPO, GeneralProperties)