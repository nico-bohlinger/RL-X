from rl_x.algorithms.ppo.flax_full_jit.default_config import get_config as get_ppo_config


def get_config(algorithm_name):
    config = get_ppo_config(algorithm_name)
    config.mpo_aux_loss_coefficient = 0.01
    config.mpo_aux_action_samples = 8
    config.mpo_aux_temperature = 1.0
    config.mpo_aux_q_critic_coef = 1.0
    config.mpo_aux_start_step = 50000000
    return config
