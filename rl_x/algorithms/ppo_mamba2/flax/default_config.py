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
    config.mamba_obs_combine_method = "concat"  # concat, film
    config.share_mamba_obs_encoder = False
    config.mamba_d_model = 128
    config.mamba_num_layers = 1
    config.mamba_expand = 2
    config.mamba_state_dim = 8
    config.mamba_conv_kernel = 4
    config.mamba_down_projection_dim = 16
    config.mamba_layer_norm_eps = 1e-5
    config.mamba_dt_min = 1e-3
    config.mamba_dt_max = 1e-1
    config.nr_hidden_units = 256
    config.action_clipping_and_rescaling = True
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
