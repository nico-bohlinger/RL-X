from rl_x.algorithms.algorithm_manager import extract_algorithm_name_from_file, register_algorithm
from rl_x.algorithms.tqc.pytorch.tqc import TQC
from rl_x.algorithms.tqc.pytorch.default_config import get_config
from rl_x.algorithms.tqc.pytorch.general_properties import GeneralProperties


TQC_PYTORCH = extract_algorithm_name_from_file(__file__)
register_algorithm(TQC_PYTORCH, get_config, TQC, GeneralProperties)
