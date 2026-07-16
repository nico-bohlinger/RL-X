from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.flashsac.pytorch.flashsac import FlashSAC
from rl_x.algorithms.flashsac.pytorch.default_config import get_config
from rl_x.algorithms.flashsac.pytorch.general_properties import GeneralProperties


FLASHSAC_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(FLASHSAC_PYTORCH, get_config, FlashSAC, GeneralProperties)
