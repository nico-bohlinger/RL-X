from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e6
    config.learning_rate_init = 3e-4
    config.learning_rate_peak = 3e-4
    config.learning_rate_end = 1.5e-4
    config.learning_rate_warmup_steps = 0
    config.buffer_size = 1000000
    config.learning_starts = 10000
    config.batch_size = 512
    config.updates_per_step = 1
    config.policy_delay = 2
    config.gamma = 0.99
    config.n_steps = 1
    config.tau = 0.01
    config.policy_hidden_dim = 128
    config.policy_nr_blocks = 2
    config.critic_hidden_dim = 256
    config.critic_nr_blocks = 2
    config.nr_atoms = 101
    config.normalized_g_max = 5.0
    config.v_min = -5.0
    config.v_max = 5.0
    config.init_entropy_coefficient = 0.01
    config.target_entropy_sigma = 0.15
    config.normalize_reward = True
    config.noise_zeta_mu = 2.0
    config.noise_zeta_max = 16
    config.logging_frequency = 1000
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
