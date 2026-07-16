from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.xqc.flax.xqc import XQC
from rl_x.algorithms.xqc.flax.default_config import get_config
from rl_x.algorithms.xqc.flax.general_properties import GeneralProperties


XQC_FLAX = extract_algorithm_name_from_file(__file__)
register_algorithm(XQC_FLAX, get_config, XQC, GeneralProperties)
