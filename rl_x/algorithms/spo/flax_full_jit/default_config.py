from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.learning_rate = 3e-4
    config.anneal_learning_rate = True
    config.nr_steps = 256
    config.nr_epochs = 10
    config.minibatch_size = 262144
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.spo_epsilon = 0.2
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.clip_value_loss = True
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.normalize_observation = True
    config.normalize_reward = True
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1
    config.evaluation_active = False

    return config
