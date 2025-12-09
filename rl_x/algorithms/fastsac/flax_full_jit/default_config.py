from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.weight_decay = 0.001
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.95
    config.batch_size = 8192
    config.buffer_size_per_env = 1024
    config.learning_starts = 10
    config.v_min = -20.0
    config.v_max = 20.0
    config.tau = 0.125
    config.gamma = 0.97
    config.nr_atoms = 101
    config.n_steps = 1
    config.target_entropy = 0.0
    config.alpha_init = 0.001
    config.log_std_min = -5.0
    config.log_std_max = 0.0
    config.nr_critic_updates_per_policy_update = 4
    config.nr_policy_updates_per_step = 2
    config.clipped_double_q_learning = False
    config.max_grad_norm = -1.0  # -1.0 to disable
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-8
    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = 18350080  # -1 to disable
    config.evaluation_active = True

    return config
