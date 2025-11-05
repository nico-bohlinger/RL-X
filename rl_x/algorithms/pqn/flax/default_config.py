from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e7
    config.learning_rate = 2.5e-4
    config.anneal_learning_rate = False
    config.nr_steps = 32
    config.nr_epochs = 2
    config.nr_minibatches = 4
    config.gamma = 0.99
    config.q_lambda = 0.65
    config.epsilon_start = 1.0
    config.epsilon_end = 0.001
    config.epsilon_decay_fraction = 0.1
    config.max_grad_norm = 10.0
    config.nr_hidden_units = 512
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10

    return config
