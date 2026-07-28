from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.xqc.pytorch.xqc import XQC
from rl_x.algorithms.xqc.pytorch.default_config import get_config
from rl_x.algorithms.xqc.pytorch.general_properties import GeneralProperties


XQC_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(XQC_PYTORCH, get_config, XQC, GeneralProperties)
