from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.sac.pytorch.sac import SAC
from rl_x.algorithms.sac.pytorch.default_config import get_config
from rl_x.algorithms.sac.pytorch.general_properties import GeneralProperties


SAC_TORCHSCRIPT = extract_algorithm_name_from_file(__file__)
register_algorithm(SAC_TORCHSCRIPT, get_config, SAC, GeneralProperties)
