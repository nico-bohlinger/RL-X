from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.flashsac.flax.flashsac import FlashSAC
from rl_x.algorithms.flashsac.flax.default_config import get_config
from rl_x.algorithms.flashsac.flax.general_properties import GeneralProperties


FLASHSAC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(FLASHSAC_FLAX, get_config, FlashSAC, GeneralProperties)
