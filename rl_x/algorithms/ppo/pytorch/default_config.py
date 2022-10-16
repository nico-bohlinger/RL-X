from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.algorithm_name = algorithm_name
    
    config.device = "cuda"  # cpu, cuda
    config.total_timesteps = 1e9
    config.nr_envs = 2
    config.learning_rate = 3e-4
    config.anneal_learning_rate = True
    config.nr_steps = 64
    config.nr_epochs = 10
    config.minibatch_size = 16
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.clip_range_vf = None
    config.ent_coef = 0.0
    config.vf_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    
    config.nr_hidden_layers = 2
    config.nr_hidden_units = 64

    config.batch_size = config.nr_envs * config.nr_steps
    config.nr_updates = config.total_timesteps / config.batch_size

    return config