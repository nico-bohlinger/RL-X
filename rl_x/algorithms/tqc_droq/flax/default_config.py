from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.algorithm_name = algorithm_name

    config.device = "cuda"  # cpu, cuda
    config.total_timesteps = 1e9
    config.nr_envs = 1
    config.learning_rate = 3e-4
    config.anneal_learning_rate = False
    config.buffer_size = 1e6
    config.learning_starts = 5000
    config.batch_size = 256
    config.tau = 0.005
    config.gamma = 0.99
    config.ensemble_size = 2
    config.nr_atoms_per_net = 25
    config.nr_dropped_atoms_per_net = 2
    config.huber_kappa = 1.0
    config.q_update_steps = 1
    config.dropout_rate = 0.01
    config.policy_update_steps = 1
    config.entropy_update_steps = 1
    config.entropy_coef = "auto"
    config.target_entropy = "auto"
    config.log_std_min = -5
    config.log_std_max = 2
    config.log_freq = 100

    config.nr_hidden_units = 256  # tqc paper uses 256 for policy and 512 for critics

    return config
