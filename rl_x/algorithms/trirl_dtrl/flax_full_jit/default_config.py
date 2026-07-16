from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 100e6
    config.learning_rate = 1e-4
    config.anneal_learning_rate = False
    config.nr_steps = 10
    config.nr_epochs = 10
    config.minibatch_size = 512
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.004
    config.critic_coef = 1.0
    config.max_grad_norm = 10.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True
    config.learning_rate_disc = 5e-04
    config.nr_epochs_disc = 10
    config.env_reward_frac = 0.0
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"
    config.epsilon = 0.2
    config.disc_buffer_capacity = 100
    config.beta = float(1/config.entropy_coef)
    config.mean_bound = 0.0001
    config.cov_bound = 0.005
    config.trust_region_coef = 0.5
    config.nr_epochs_rew = 30
    config.learning_rate_reward_fn = 5e-05
    config.gp_lambda = 0.05
    config.gp_alpha = 0.5
    config.handle_absorbing_states = True
    config.reward_fn_approximator = False
    config.on_demand_etas = Falsee

    return config
