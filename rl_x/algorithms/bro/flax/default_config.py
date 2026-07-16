from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e6
    config.policy_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.entropy_coefficient_learning_rate = 3e-4
    config.adjustment_learning_rate = 3e-5
    config.buffer_size = 1000000
    config.learning_starts = 2500
    config.batch_size = 128
    config.updates_per_step = 10
    config.gamma = 0.99
    config.tau = 5e-3
    config.policy_hidden_dim = 256
    config.policy_nr_blocks = 1
    config.critic_hidden_dim = 512
    config.critic_nr_blocks = 2
    config.distributional = True
    config.nr_quantiles = 100
    config.pessimism = 0.0
    config.kl_target = 0.05
    config.std_multiplier = 0.75
    config.init_entropy_coefficient = 1.0
    config.init_optimism = 1.0
    config.init_regularizer = 0.25
    config.target_entropy = "auto"
    config.use_optimistic_exploration = True
    config.first_reset_step = 15000
    config.reset_interval = 500000
    config.logging_frequency = 1000
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
