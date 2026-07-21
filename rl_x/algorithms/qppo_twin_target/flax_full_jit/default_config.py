from rl_x.algorithms.qppo.flax_full_jit.default_config import get_config as get_qppo_config


def get_config(algorithm_name):
    config = get_qppo_config(algorithm_name)
    config.nr_q_critics = 2
    config.critic_tau = 0.005
    return config
