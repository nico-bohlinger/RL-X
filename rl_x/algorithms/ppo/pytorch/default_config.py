from ml_collections import config_dict
import time


def get_config(algorithm, environment):
    config = config_dict.ConfigDict()

    config.algorithm_name = algorithm.name

    config.tb_track = False
    config.wandb_track = False
    config.project_name = "placeholder"
    config.exp_name = "placeholder"
    config.run_name = f"{int(time.time())}"
    config.run_path = f"runs/{config.project_name}/{config.exp_name}/{config.run_name}"

    config.mode = "train"  # train, test
    config.seed = 1

    config.device = "cuda"  # cpu, cuda
    config.total_timesteps = 1e9
    config.nr_envs = 2
    config.learning_rate = 3e-4
    config.anneal_learning_rate = True
    config.nr_steps = 64
    config.nr_epochs = 10
    config.minibatch_size = 16
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.clip_range_vf = None
    config.ent_coef = 0.0
    config.vf_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0

    config.batch_size = config.nr_envs * config.nr_steps
    config.nr_updates = config.total_timesteps / config.batch_size

    return config