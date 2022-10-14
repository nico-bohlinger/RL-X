import random
import numpy as np
import jax


class SAC():
    def __init__(self, config, env, writer, rlx_logger) -> None:
        self.config = config
        self.env = env
        self.writer = writer
        self.rlx_logger = rlx_logger
        
        self.seed = config.algorithm.seed
        self.device = config.algorithm.device
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

        if self.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        key = jax.random.PRNGKey(self.seed)
    

    def train(self):
        pass


    def test(self):
        pass
