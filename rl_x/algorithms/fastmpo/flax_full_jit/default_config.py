from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 2000158720
    config.critic_network_type = "fastsac"  # fastsac, fasttd3, mpo
    config.dual_critic = True
    config.policy_network_type = "fastsac"  # fastsac, fasttd3, mpo
    config.action_clipping = True
    config.action_rescaling = "fastsac"  # "none", "fastsac", "normal"
    config.policy_learning_rate = 3e-4
    config.critic_learning_rate = 3e-4
    config.dual_learning_rate = 1e-2
    config.anneal_policy_learning_rate = False
    config.anneal_critic_learning_rate = False
    config.anneal_dual_learning_rate = False
    config.policy_weight_decay = 0.001
    config.critic_weight_decay = 0.001
    config.dual_weight_decay = 0.0
    config.adam_beta1 = 0.9  # default: 0.9
    config.adam_beta2 = 0.95  # default: 0.999
    config.max_grad_norm = 40.0
    config.collect_data_with_online_policy = False
    config.action_sampling_number = 20
    config.epsilon_non_parametric = 0.1
    config.epsilon_parametric_mu = 0.01
    config.epsilon_parametric_sigma = 1e-6
    config.epsilon_penalty = 0.001
    config.init_log_eta = 10.0
    config.init_log_alpha_mean = 10.0
    config.init_log_alpha_stddev = 1000.0
    config.init_log_penalty_temperature = 10.0
    config.float_epsilon = 1e-8
    config.min_log_temperature = -18.0
    config.min_log_alpha = -18.0
    config.policy_init_scale = 0.5
    config.policy_min_scale = 1e-6
    config.batch_size = 8192  # fastsac: 8192, fasttd3: 32768
    config.buffer_size_per_env = 1024  # fastsac: 1024, fasttd3: 10240
    config.learning_starts = 10  # times nr_envs
    config.v_min = -20.0  # fastsac: -20.0, fasttd3: -10.0
    config.v_max = 20.0  # fastsac: 20.0, fasttd3: 10.0
    config.critic_tau = 0.125  # fastsac: 0.125, fasttd3: 0.1
    config.policy_tau = 0.01
    config.gamma = 0.97
    config.nr_atoms = 101
    config.n_steps = 1
    config.clipped_double_q_learning = False  # only possible if dual_critic is True
    config.nr_critic_updates_per_policy_update = 4  # fastsac: 4, fasttd3: 2, mpo: 1
    config.nr_policy_updates_per_step = 2  # fastsac: 2, fasttd3: 1, mpo: some
    config.enable_observation_normalization = True
    config.normalizer_epsilon = 1e-8
    config.logging_frequency = 40960
    config.evaluation_and_save_frequency = 18350080  # -1 to disable
    config.evaluation_active = True

    return config
