from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"  # cpu, gpu, mps
    config.compile_mode = "default"
    config.bf16_mixed_precision_training = False
    config.total_timesteps = 1e9
    config.policy_learning_rate = 1e-4
    config.critic_learning_rate = 1e-3
    config.anneal_learning_rate = False
    config.nr_steps = 24
    config.nr_epochs = 10
    config.minibatch_size = 49152
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clipping_epsilon = 0.1
    config.clipping_epsilon_base = 0.1
    config.clipping_epsilon_rate = 3.0
    config.critic_coef = 0.5
    config.max_grad_norm = -1.0
    config.target_kl = 1.0
    config.reward_scaling = 1.0
    config.normalize_reward = True
    config.reward_clip = 10.0
    config.normalize_observation = True
    config.action_rescaling = True
    config.diffusion_steps = 10
    config.timestep_embed_dim = 16
    config.policy_hidden_dims = (512, 512, 512)
    config.critic_hidden_dims = (256, 256, 256)
    config.policy_output_scale = 1.0
    config.denoising_std = 0.1
    config.denoising_discount = 1.0
    config.denoised_clip_value = 1.0
    config.noise_clip_value = 3.0
    config.log_probability_min = -5.0
    config.log_probability_max = 2.0
    config.advantage_quantile_min = 0.05
    config.advantage_quantile_max = 0.95
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
