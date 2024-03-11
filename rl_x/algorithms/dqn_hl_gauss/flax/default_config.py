from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.buffer_size = 1e5
    config.learning_starts = 20000
    config.batch_size = 32
    config.gamma = 0.99
    config.nr_bins = 50
    config.sigma_to_final_sigma_ratio = 0.75
    config.v_min = -10
    config.v_max = 10
    config.epsilon_start = 1.0
    config.epsilon_end = 0.01
    config.epsilon_decay_steps = 250000
    config.update_frequency = 4
    config.target_update_frequency = 8000
    config.nr_hidden_units = 512
    config.logging_frequency = 1000
    config.evaluation_frequency = 200000  # -1 to disable
    config.evaluation_episodes = 10

    return config
