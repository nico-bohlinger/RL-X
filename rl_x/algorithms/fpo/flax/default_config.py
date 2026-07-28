from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name
    config.device = "gpu"
    config.total_timesteps = 1e9
    config.learning_rate = 1e-4
    config.weight_decay = 1e-4
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.anneal_learning_rate = False
    config.nr_steps = 24
    config.nr_epochs = 32
    config.minibatch_size = 24576
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clipping_epsilon = 0.05
    config.critic_coef = 1.0
    config.max_grad_norm = 1.0
    config.reward_scaling = 1.0
    config.normalize_observation = True
    config.observation_normalizer_epsilon = 1e-2
    config.observation_normalizer_max_count = 100000000
    config.flow_steps = 64
    config.timestep_embed_dim = 8
    config.policy_hidden_dims = (256, 256, 256)
    config.critic_hidden_dims = (768, 768, 768)
    config.actor_scale = 1.0
    config.policy_output_scale = 1.0
    config.action_clip = 2.0
    config.nr_flow_samples_per_action = 32
    config.timestep_inverse_cdf_beta = 1.0
    config.action_perturb_std = 0.02
    config.cfm_loss_clamp = 20.0
    config.cfm_loss_clamp_negative_advantages_max = 20.0
    config.cfm_difference_clamp_max = 10.0
    config.trust_region_mode = "aspo"
    config.advantage_clamp = 100.0
    config.ema_decay = 0.95
    config.ema_warmup_steps = 500
    config.evaluation_frequency = -1
    config.evaluation_episodes = 10

    return config
