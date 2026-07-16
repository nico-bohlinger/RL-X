from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.nr_parallel_seeds = 1
    config.total_timesteps = 100e6
    config.learning_rate = 4e-05
    config.anneal_learning_rate = False
    config.nr_steps = 10
    config.nr_epochs = 10
    config.minibatch_size = 512
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.entropy_coef = 0.001
    config.critic_coef = 1.0
    config.max_grad_norm = 10.0
    config.std_dev = 1.0
    config.action_clipping_and_rescaling = False
    config.evaluation_and_save_frequency = -1  # -1 to disable
    config.evaluation_active = True
    config.learning_rate_disc = 1e-05
    config.nr_epochs_disc = 10
    config.env_reward_frac = 0.0
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"

    config.gp_lambda = 0.05
    config.gp_alpha = 0.5
    config.handle_absorbing_states = True

    return config
