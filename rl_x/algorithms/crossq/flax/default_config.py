from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.learning_rate = 1e-3
    config.anneal_learning_rate = False
    config.policy_adam_b1 = 0.5
    config.critic_adam_b1 = 0.5
    config.buffer_size = 10e6
    config.learning_starts = 5000
    config.batch_size = 256
    config.gamma = 0.99
    config.ensemble_size = 2
    config.policy_delay = 3
    config.target_entropy = "auto"
    config.log_std_min = -20
    config.log_std_max = 2
    config.batch_renorm_momentum = 0.99
    config.batch_renorm_warmup_steps = 100000
    config.policy_nr_hidden_units = 256
    config.critic_nr_hidden_units = 2048
    config.logging_frequency = 3000
    config.evaluation_frequency = 200000  # -1 to disable
    config.evaluation_episodes = 10

    return config
