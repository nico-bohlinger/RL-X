from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2_000_000_000
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.nr_steps = 128
    config.nr_epochs = 8
    config.minibatch_size = 2048
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clipping_epsilon = 0.2
    config.critic_coef = 0.25
    config.max_grad_norm = 1.0
    config.reward_scaling = 10.0
    config.normalize_observation = True

    config.flow_steps = 10
    config.timestep_embed_dim = 8
    config.policy_hidden_dims = (32, 32, 32, 32)
    config.critic_hidden_dims = (256, 256, 256, 256, 256)
    config.policy_output_scale = 0.25
    config.output_mode = "u_but_supervise_as_eps"
    config.nr_flow_samples_per_action = 8
    config.average_losses_before_exp = True
    config.discretize_t_for_training = True
    config.feather_std = 0.0

    config.evaluation_and_save_frequency = 16_777_216
    config.evaluation_active = False

    return config
