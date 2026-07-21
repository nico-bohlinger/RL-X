from rl_x.algorithms.qppo_twin_target.flax_full_jit.default_config import get_config as get_qppo_twin_target_config


def get_config(algorithm_name):
    config = get_qppo_twin_target_config(algorithm_name)
    config.q_critic_reduction = "mean"
    return config
