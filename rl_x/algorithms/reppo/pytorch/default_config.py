from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu, mps
    config.compile_mode = "reduce-overhead"  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
    config.bf16_mixed_precision_training = True
    config.total_timesteps = 1000000000
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.nr_steps = 128
    config.nr_epochs = 4
    config.nr_minibatches = 128
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.max_grad_norm = 0.5
    config.policy_hidden_dim = 512
    config.critic_hidden_dim = 512
    config.policy_min_std = 0.0
    config.nr_bins = 151
    config.v_min = -100.0
    config.v_max = 100.0
    config.init_kl_coefficient = 0.01
    config.kl_bound = 0.1
    config.init_entropy_coefficient = 0.01
    config.target_entropy_multiplier = 0.5
    config.auxiliary_loss_coefficient = 1.0
    config.nr_kl_samples = 16
    config.normalize_observation = True
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
