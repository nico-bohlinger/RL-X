from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.learning_rate = 3e-4
    config.anneal_learning_rate = True
    config.weight_decay = 0.1
    config.batch_size = 32768
    config.buffer_size_per_env = 5120  # 1024 * 5
    config.learning_starts = 10
    config.v_min = -10.0
    config.v_max = 10.0
    config.tau = 0.1
    config.gamma = 0.97
    config.nr_atoms = 101
    config.n_steps = 8
    config.noise_std_min = 0.05
    config.noise_std_max = 0.8
    config.smoothing_epsilon = 0.001
    config.smoothing_clip_value = 0.5
    config.nr_critic_updates_per_policy_update = 2
    config.nr_policy_updates_per_step = 2
    config.clipped_double_q_learning = True
    config.max_grad_norm = 0.0
    config.action_clipping_and_rescaling = False
    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = 18350080  # -1 to disable
    config.evaluation_active = True

    return config
