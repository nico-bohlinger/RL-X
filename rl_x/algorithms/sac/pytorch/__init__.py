from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.sac.pytorch.sac import SAC
from rl_x.algorithms.sac.pytorch.default_config import get_config


algorithm_name = __file__.split("rl_x/algorithms/")[1].split("/__init__.py")[0].replace("/", ".")

SAC_PYTORCH = algorithm_name
register_algorithm(SAC_PYTORCH, get_config, SAC)
