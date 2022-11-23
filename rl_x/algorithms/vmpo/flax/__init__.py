from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.vmpo.flax.vmpo import VMPO
from rl_x.algorithms.vmpo.flax.default_config import get_config


VMPO_FLAX = "V-MPO Flax"
register_algorithm(VMPO_FLAX, get_config, VMPO)
