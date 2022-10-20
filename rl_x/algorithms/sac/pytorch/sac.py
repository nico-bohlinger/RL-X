import os
import logging
import random
import numpy as np
import torch

rlx_logger = logging.getLogger("rl_x")


class SAC():
    def __init__(self, config, env, writer):
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(config.runner.run_path, "models")
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.algorithm.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.train_freq = config.algorithm.train_freq
        self.gradient_steps = config.algorithm.gradient_steps
        self.ent_coef = config.algorithm.ent_coef
        self.target_update_interval = config.algorithm.target_update_interval
        self.target_entropy = config.algorithm.target_entropy
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        self.device = torch.device("cuda" if config.algorithm.device == "cuda" and torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape






        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
