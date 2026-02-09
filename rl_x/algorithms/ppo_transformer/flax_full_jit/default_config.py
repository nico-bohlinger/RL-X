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
    config.tf_obs_combine_method = "concat"  # concat, film
    config.share_tf_obs_encoder = False
    config.tf_context_len = 16
    config.tf_d_model = 128
    config.tf_dim_feedforward = 512
    config.tf_down_projection_dim = 16
    config.tf_nhead = 4
    config.tf_num_layers = 2
    config.tf_dropout = 0.0
    config.tf_layer_norm_eps = 1e-5
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = 17301504  # -1 to disable
    config.evaluation_active = True

    return config
