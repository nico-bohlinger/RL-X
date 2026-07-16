from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e6
    config.learning_rate_init = 1e-4
    config.learning_rate_end = 5e-5
    config.buffer_size = 1000000
    config.learning_starts = 5000
    config.batch_size = 256
    config.updates_per_step = 2
    config.gamma = 0.99
    config.tau = 5e-3
    config.policy_hidden_dim = 128
    config.policy_nr_blocks = 1
    config.critic_hidden_dim = 512
    config.critic_nr_blocks = 2
    config.c_shift = 3.0
    config.use_cdq = True
    config.nr_bins = 101
    config.v_min = -5.0
    config.v_max = 5.0
    config.log_std_min = -10.0
    config.log_std_max = 2.0
    config.init_entropy_coefficient = 1e-2
    config.target_entropy = "auto"
    config.normalize_observation = True
    config.normalize_reward = True
    config.normalized_g_max = 5.0
    config.logging_frequency = 1000
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
