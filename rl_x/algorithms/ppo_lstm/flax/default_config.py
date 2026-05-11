from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.nr_steps = 64
    config.nr_epochs = 10
    config.minibatch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.obs_encoding_dim = 128
    config.lstm_hidden_dim = 64
    config.lstm_obs_combine_method = "concat"  # concat, film
    config.share_lstm_obs_encoder = False
    config.action_clipping_and_rescaling = True
    config.nr_hidden_units = 256
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10

    return config
