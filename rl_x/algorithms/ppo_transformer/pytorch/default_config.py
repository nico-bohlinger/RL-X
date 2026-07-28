from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu, mps
    config.compile_mode = "default"  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
    config.bf16_mixed_precision_training = False
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
    config.nr_hidden_units = 256
    config.action_clipping_and_rescaling = True
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
