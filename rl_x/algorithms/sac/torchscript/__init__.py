from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.torchscript.sac import SAC
from rl_x.algorithms.sac.torchscript.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

SAC_TORCHSCRIPT = algorithm_name
register_algorithm(SAC_TORCHSCRIPT, get_config, SAC)
