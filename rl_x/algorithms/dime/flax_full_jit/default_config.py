from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2_000_158_720
    config.actor_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.entropy_learning_rate = 1e-3
    config.adam_beta1 = 0.5
    config.adam_beta2 = 0.999
    config.batch_size = 8192
    config.buffer_size_per_env = 1024
    config.learning_starts = 10
    config.updates_per_step = 2
    config.policy_delay = 3
    config.gamma = 0.99
    config.critic_tau = 1.0
    config.policy_tau = 1.0
    config.critic_hidden_dims = (2048, 2048)
    config.nr_critics = 2
    config.nr_atoms = 101
    config.v_min = -200.0
    config.v_max = 200.0
    config.critic_entropy_coefficient = 0.005

    config.diffusion_steps = 16
    config.score_hidden_dims = (256, 256, 256)
    config.timestep_embed_dim = 32
    config.prior_std = 2.5
    config.minimum_timestep = 0.001
    config.cosine_schedule_offset = 0.008
    config.score_output_scale = 1e-8
    config.entropy_coefficient_init = 1.0
    config.target_entropy_per_action_dimension = 4.0
    config.max_grad_norm = 1.0
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-8

    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = 18_350_080
    config.evaluation_active = False

    return config
