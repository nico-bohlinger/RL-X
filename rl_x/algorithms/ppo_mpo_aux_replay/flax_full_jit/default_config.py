from rl_x.algorithms.ppo_mpo_aux.flax_full_jit.default_config import get_config as get_ppo_mpo_aux_config


def get_config(algorithm_name):
    config = get_ppo_mpo_aux_config(algorithm_name)
    config.mpo_aux_replay_buffer_size_per_env = 256
    config.mpo_aux_replay_batch_size = 8192
    config.mpo_aux_replay_critic_tau = 0.005
    return config
