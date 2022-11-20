from rl_x.algorithms.algorithm_manager import register_algorithm
from rl_x.algorithms.tqc_qdivq.flax.tqc_qdivq import TQC_QdivQ
from rl_x.algorithms.tqc_qdivq.flax.default_config import get_config


TQC_QDIVQ_FLAX = "TQC+QdivQ Flax"
register_algorithm(TQC_QDIVQ_FLAX, get_config, TQC_QdivQ)
