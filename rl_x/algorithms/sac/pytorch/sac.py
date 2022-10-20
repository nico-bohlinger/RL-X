import os
import logging
import random
import numpy as np
import torch
import torch.optim as optim

from rl_x.algorithms.sac.pytorch.actor import get_actor
from rl_x.algorithms.sac.pytorch.critic import get_critic
from rl_x.algorithms.sac.pytorch.replay_buffer import ReplayBuffer

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
        self.target_entropy = config.algorithm.target_entropy
        self.target_update_interval = config.algorithm.target_update_interval
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        self.device = torch.device("cuda" if config.algorithm.device == "cuda" and torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        self.actor = get_actor(config, env, self.device).to(self.device)
        self.q1 = get_critic(config, env).to(self.device)
        self.q2 = get_critic(config, env).to(self.device)
        self.q1_target = get_critic(config, env).to(self.device)
        self.q2_target = get_critic(config, env).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.learning_rate)

        if self.ent_coef == "auto":
            if self.target_entropy == "auto":
                self.target_entropy = -torch.prod(torch.tensor(env.get_single_action_space_shape(), dtype=torch.float32).to(self.device)).item()
                log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha = log_alpha.exp()
                self.alpha_optimizer = optim.Adam([log_alpha], lr=self.learning_rate)
            
        else:
            self.alpha = self.ent_coef

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.os_shape, self.as_shape, self.device)
        exit()
    

    def set_train_mode(self):
        self.actor.train()
        self.q1.train()
        self.q2.train()
        self.q1_target.train()
        self.q2_target.train()


    def set_eval_mode(self):
        self.actor.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
