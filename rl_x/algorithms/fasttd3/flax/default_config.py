from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.buffer_size = 1e6
    config.learning_starts = 5000
    config.batch_size = 256
    config.tau = 0.005
    config.gamma = 0.99
    config.epsilon = 0.1
    config.smoothing_epsilon = 0.2
    config.smoothing_clip_value = 0.5
    config.policy_delay = 2
    config.nr_hidden_units = 256
    config.logging_frequency = 3000
    config.evaluation_frequency = 200000  # -1 to disable
    config.evaluation_episodes = 10

    return config
