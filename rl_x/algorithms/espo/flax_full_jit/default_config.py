from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.learning_rate = 3e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.max_epochs = 20
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.max_ratio_delta = 0.25
    config.delta_calc_operator = "mean"
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1
    config.evaluation_active = False

    return config
