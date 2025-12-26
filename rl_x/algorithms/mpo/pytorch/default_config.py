from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu, mps
    config.compile_mode = "reduce-overhead"  # default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs
    config.bf16_mixed_precision_training = True
    config.total_timesteps = 1e9
    config.agent_learning_rate = 3e-4
    config.dual_learning_rate = 1e-2
    config.anneal_agent_learning_rate = False
    config.anneal_dual_learning_rate = False
    config.buffer_size = 1e6
    config.learning_starts = 5000
    config.batch_size = 256
    config.actor_update_period = 1000
    config.target_network_update_period = 100
    config.tau = 0.005
    config.gamma = 0.99
    config.n_steps = 4
    config.optimize_every_n_steps = 4
    config.action_sampling_number = 20
    config.grad_norm_clip = 40.0
    config.epsilon_non_parametric = 0.1
    config.epsilon_parametric_mu = 0.01
    config.epsilon_parametric_sigma = 1e-6
    config.epsilon_penalty = 0.001
    config.init_log_eta = 10.0
    config.init_log_alpha_mean = 10.0
    config.init_log_alpha_stddev = 1000.0
    config.init_log_penalty_temperature = 10.0
    config.policy_init_scale = 0.5
    config.policy_min_scale = 1e-6
    config.action_rescaling = False
    config.v_min = -1600.0
    config.v_max = 1600.0
    config.nr_atoms = 51
    config.nr_hidden_units = 256
    config.float_epsilon = 1e-8
    config.min_log_temperature = -18.0
    config.min_log_alpha = -18.0
    config.logging_frequency = 300
    config.evaluation_frequency = 200000  # -1 to disable
    config.evaluation_episodes = 10

    return config
