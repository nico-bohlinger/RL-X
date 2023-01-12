from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.algorithm_name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.agent_learning_rate = 3e-4
    config.dual_learning_rate = 1e-2
    config.anneal_agent_learning_rate = False
    config.anneal_dual_learning_rate = False
    config.buffer_size = 1e6
    config.learning_starts = 5000
    config.batch_size = 256
    config.tau = 0.005
    config.gamma = 0.99
    config.nr_samples = 20
    config.init_log_temperature = 10.0
    config.init_log_alpha_mean = 10.0
    config.init_log_alpha_stddev = 1000.0
    config.min_log_temperature = -18.0
    config.min_log_alpha = -18.0
    config.kl_epsilon = 0.1
    config.kl_epsilon_penalty = 0.001
    config.kl_epsilon_mean = 0.005  # Acme docu mentions: Divide it by nr action space dimension, orig mean = 0.1
    config.kl_epsilon_stddev = 1e-6  # Acme docu mentions: Divide it by nr action space dimension, orig stddev = 0.0001
    config.retrace_lambda = 0.95
    config.trace_length = 8
    config.init_stddev = 0.5
    config.min_stddev = 1e-6
    config.stability_epsilon = 1e-8
    config.nr_hidden_units = 256
    config.logging_all_metrics = False  # Some metrics seem to cost performance when logged, so they get only logged if True
    config.logging_freq = 3000

    return config
