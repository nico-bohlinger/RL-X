from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.critic_learning_rate = 1e-3
    config.anneal_critic_learning_rate = False
    config.nr_steps = 128
    config.critic_minibatch_size = 32768
    config.nr_critic_updates = 10
    config.gamma = 0.99
    config.gae_lambda = 0.97
    config.target_kl = 0.01
    config.cg_max_steps = 10
    config.cg_damping = 0.1
    config.cg_residual_tolerance = 1e-10
    config.line_search_shrinking_factor = 0.8
    config.line_search_max_steps = 10
    config.policy_subsampling_factor = 1
    config.critic_max_grad_norm = 5.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1
    config.evaluation_active = False

    return config
