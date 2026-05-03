from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2e9
    config.learning_rate = 4e-4
    config.anneal_learning_rate = True
    config.nr_steps = 128
    config.nr_epochs = 10
    config.minibatch_size = 32768
    config.gamma = 0.99
    config.gae_lambda = 0.9
    config.clip_range = 0.1
    config.entropy_coef = 0.0
    config.critic_coef = 1.0
    config.max_grad_norm = 5.0
    config.std_dev = 1.0
    config.mamba_obs_combine_method = "concat"  # concat, film
    config.share_mamba_obs_encoder = False
    config.mamba_d_model = 128
    config.mamba_num_layers = 2
    config.mamba_expand = 2
    config.mamba_state_dim = 16
    config.mamba_conv_kernel = 4
    config.mamba_down_projection_dim = 32
    config.mamba_layer_norm_eps = 1e-5
    config.mamba_dt_min = 1e-3
    config.mamba_dt_max = 1e-1
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    return config
