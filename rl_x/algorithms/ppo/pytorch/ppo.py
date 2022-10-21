import os
import random
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from rl_x.algorithms.ppo.pytorch.actor import get_actor
from rl_x.algorithms.ppo.pytorch.critic import get_critic
from rl_x.algorithms.ppo.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class PPO:
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
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.clip_range_vf = config.algorithm.clip_range_vf
        self.ent_coef = config.algorithm.ent_coef
        self.vf_coef = config.algorithm.vf_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_layers = config.algorithm.nr_hidden_layers
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.batch_size = config.algorithm.nr_envs * config.algorithm.nr_steps

        self.device = torch.device("cuda" if config.algorithm.device == "cuda" and torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        self.actor = get_actor(config, env, self.device).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.critic = get_critic(config, env).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        self.set_train_mode()

        batch = Batch(
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32).to(self.device),
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            values = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            dones = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            log_probs = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device)
        )
        
        state = torch.tensor(self.env.reset(), dtype=torch.float32).to(self.device)
        saving_return_buffer = deque(maxlen=100)
        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
        

            # Acting
            episode_info_buffer = deque(maxlen=100)
            for step in range(self.nr_steps):
                with torch.no_grad():
                    action, processed_action, log_prob = self.actor.get_action_logprob(state)
                    value = self.critic.get_value(state)
                next_state, reward, done, info = self.env.step(processed_action.cpu().numpy())
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                actual_next_state = next_state.clone()
                for i, single_done in enumerate(done):
                    if single_done:
                        maybe_terminal_observation = self.env.get_terminal_observation(info, i)
                        if maybe_terminal_observation is not None:
                            actual_next_state[i] = torch.tensor(maybe_terminal_observation, dtype=torch.float32).to(self.device)

                batch.states[step] = state
                batch.actions[step] = action
                batch.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                batch.values[step] = value.reshape(-1)
                batch.dones[step]= torch.tensor(done, dtype=torch.float32).to(self.device)
                batch.log_probs[step] = log_prob     
                state = next_state
                global_step += self.nr_envs

                episode_info_buffer.extend(self.env.get_episode_infos(info))
                if len(episode_info_buffer) > 0:
                    ep_info_returns = [ep_info["r"] for ep_info in episode_info_buffer]
                    saving_return_buffer.extend(ep_info_returns)
            
            acting_end_time = time.time()


            # Calculating advantages and returns
            with torch.no_grad():
                next_value = self.critic.get_value(actual_next_state).reshape(1, -1)
            advantages = torch.zeros_like(batch.rewards, dtype=torch.float32).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.nr_steps)):
                if t == self.nr_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = batch.values[t + 1]
                not_done = 1.0 - batch.dones[t]
                delta = batch.rewards[t] + self.gamma * nextvalues * not_done - batch.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * not_done * lastgaelam
            returns = advantages + batch.values
            
            calc_adv_return_end_time = time.time()


            # Optimizing
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                for param_group in self.actor_optimizer.param_groups + self.critic_optimizer.param_groups:
                    param_group["lr"] = learning_rate
            
            batch_states = batch.states.reshape((-1,) + self.os_shape)
            batch_actions = batch.actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_values = batch.values.reshape(-1)
            batch_log_probs = batch.log_probs.reshape(-1)

            pg_losses = []
            value_losses = []
            entropy_losses = []
            loss_losses = []
            clip_fractions = []

            batch_indices = np.arange(self.batch_size)
            for epoch in range(self.nr_epochs):
                approx_kl_divs = []
                np.random.shuffle(batch_indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    new_log_prob, entropy = self.actor.get_logprob_entropy(batch_states[minibatch_indices], batch_actions[minibatch_indices])
                    new_value = self.critic.get_value(batch_states[minibatch_indices])
                    logratio = new_log_prob - batch_log_probs[minibatch_indices]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        log_ratio = new_log_prob - batch_log_probs[minibatch_indices]
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    minibatch_advantages = batch_advantages[minibatch_indices]
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                    pg_loss1 = -minibatch_advantages * ratio
                    pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                    new_value = new_value.reshape(-1)
                    v_loss1 = 0.5 * (new_value - batch_returns[minibatch_indices]) ** 2
                    if self.clip_range_vf != None:
                        v_clipped = batch_values[minibatch_indices] + torch.clamp(new_value - batch_values[minibatch_indices], -self.clip_range_vf, self.clip_range_vf)
                        v_loss2 = 0.5 * (v_clipped - batch_returns[minibatch_indices]) ** 2
                        v_loss = torch.maximum(v_loss1, v_loss2).mean()
                    else:
                        v_loss = v_loss1.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy_loss

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    pg_losses.append(pg_loss.item())
                    value_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    loss_losses.append(loss.item())
                    clip_fractions.append(torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item())

            y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            optimizing_end_time = time.time()


            # Saving
            # Only save when the total return buffer (over multiple updates) isn't empty
            # Also only save when the episode info buffer isn't empty -> there were finished episodes this update
            if self.save_model and saving_return_buffer and episode_info_buffer:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()


            # Logging
            if self.track_tb:
                if len(episode_info_buffer) > 0:
                    self.writer.add_scalar("rollout/ep_rew_mean", np.mean(ep_info_returns), global_step)
                    self.writer.add_scalar("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.writer.add_scalar(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                self.writer.add_scalar("time/fps", int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time)), global_step)
                self.writer.add_scalar("time/acting_time", acting_end_time - start_time, global_step)
                self.writer.add_scalar("time/calc_advantages_and_return_time", calc_adv_return_end_time - acting_end_time, global_step)
                self.writer.add_scalar("time/optimizing_time", optimizing_end_time - calc_adv_return_end_time, global_step)
                self.writer.add_scalar("time/saving_time", saving_end_time - optimizing_end_time, global_step)
                self.writer.add_scalar("train/learning_rate", learning_rate, global_step)
                self.writer.add_scalar("train/clip_range", self.clip_range, global_step)
                self.writer.add_scalar("train/clip_fraction", np.mean(clip_fractions), global_step)
                self.writer.add_scalar("train/approx_kl", np.mean(approx_kl_divs), global_step)
                self.writer.add_scalar("train/policy_gradient_loss", np.mean(pg_losses), global_step)
                self.writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
                self.writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), global_step)
                self.writer.add_scalar("train/loss", np.mean(loss_losses), global_step)
                self.writer.add_scalar("train/std", np.mean(np.exp(self.actor.actor_logstd.data.cpu().numpy())), global_step)
                self.writer.add_scalar("train/explained_variance", explained_var, global_step)

            rlx_logger.info(f"Step: {global_step}")


    def save(self):
        file_path = self.save_path + "/model_best.pt"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, writer):
        checkpoint = torch.load(config.runner.load_model)
        config.algorithm = checkpoint["config_algorithm"]
        model = PPO(config, env, writer)
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        return model


    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = self.actor.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, done, info = self.env.step(processed_action.cpu().numpy())
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")


    def set_train_mode(self):
        self.actor.train()
        self.critic.train()


    def set_eval_mode(self):
        self.actor.eval()
        self.critic.eval()

