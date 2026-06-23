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
    config.minibatch_size = 64
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

    # NCSN Params
    config.batch_size_ncsn = 256
    config.minibatch_size_ncsn = 64
    config.total_samples_ncsn = 25e6
    config.nr_epochs_ncsn = 20 # Number of ncsn epochs
    config.anneal_power_ncsn = 2.0
    config.sigma_begin_ncsn = 10.0
    config.sigma_end_ncsn = 0.01
    config.L_ncsn = 20
    config.nr_hidden_units_encoder_ncsn = [256, 512, 1024, 2048]
    config.nr_hidden_units_decoder_ncsn = [1024, 512, 128, 64, 32]
    config.learning_rate_ncsn = 1e-04
    config.sigma_inference_ncsn = 5
    config.env_reward_frac = 0.0
    config.handle_absorbing_states = True
    config.state_based = False
    config.ncsnv1 = True
    config.data_path = "../trirl_dataset/rl_expert/Ant-v5_30_PPO.npz"


    return config
