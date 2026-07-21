from rl_x.algorithms.ppo.flax_full_jit.default_config import get_config as get_ppo_config


def get_config(algorithm_name):
    config = get_ppo_config(algorithm_name)
    config.nr_value_samples = 16
    return config
