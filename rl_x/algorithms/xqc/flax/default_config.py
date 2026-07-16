from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1000000
    config.policy_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.entropy_coefficient_learning_rate = 3e-4
    config.learning_rate_end = 3e-5
    config.weight_decay = 0.0
    config.buffer_size = 1000000
    config.learning_starts = 5000
    config.batch_size = 256
    config.updates_per_step = 2
    config.policy_delay = 3
    config.target_net_update_freq = 1
    config.gamma = 0.99
    config.tau = 0.005
    config.policy_hidden_dim = 256
    config.policy_nr_blocks = 4
    config.critic_hidden_dim = 512
    config.critic_nr_blocks = 4
    config.nr_critics = 2
    config.nr_atoms = 101
    config.v_min = -5.0
    config.v_max = 5.0
    config.skip_connections = False
    config.use_weight_norm = True
    config.normalize_last_layer = True
    config.decay_bn = False
    config.log_std_min = -10.0
    config.log_std_max = 2.0
    config.init_entropy_coefficient = 0.01
    config.target_entropy = "auto"
    config.normalize_reward = True
    config.logging_frequency = 1000
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
