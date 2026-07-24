from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2_000_158_720
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

    config.diffusion_steps = 8
    config.timestep_embed_dim = 8
    config.policy_hidden_dims = (256, 256, 256)
    config.critic_hidden_dims = (256, 256, 256)
    config.policy_output_scale = 1.0
    config.denoising_std = 0.1
    config.final_steps_only = False

    config.evaluation_and_save_frequency = 16_777_216
    config.evaluation_active = False

    return config
