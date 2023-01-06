import os
import random
import logging
import time
from collections import deque
import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from rl_x.algorithms.espo.torchscript.policy import get_policy
from rl_x.algorithms.espo.torchscript.critic import get_critic
from rl_x.algorithms.espo.torchscript.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class ESPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.max_epochs = config.algorithm.max_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_ratio_delta = config.algorithm.max_ratio_delta
        self.ent_coef = config.algorithm.ent_coef
        self.vf_coef = config.algorithm.vf_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps

        if config.algorithm.delta_calc_operator == "mean":
            self.delta_calc_operator = torch.mean
        elif config.algorithm.delta_calc_operator == "median":
            self.delta_calc_operator = torch.median
        else:
            raise ValueError("Unknown delta_calc_operator")

        self.device = torch.device("cuda" if config.algorithm.device == "gpu" and torch.cuda.is_available() else "cpu")
        rlx_logger.info(f"Using device: {self.device}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        self.policy = get_policy(config, env, self.device).to(self.device)
        self.critic = get_critic(config, env).to(self.device)
        self.policy = torch.jit.script(self.policy)
        self.critic = torch.jit.script(self.critic)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        self.set_train_mode()

        batch = Batch(
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            next_states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32).to(self.device),
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            values = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            dones = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            log_probs = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device)
        )

        state = torch.tensor(self.env.reset(), dtype=torch.float32).to(self.device)
        saving_return_buffer = deque(maxlen=100)
        rng = default_rng()
        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
        

            # Acting
            episode_info_buffer = deque(maxlen=100)
            for step in range(self.nr_steps):
                with torch.no_grad():
                    action, processed_action, log_prob = self.policy.get_action_logprob(state)
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
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                batch.values[step] = value.reshape(-1)
                batch.dones[step]= torch.tensor(done, dtype=torch.float32).to(self.device)
                batch.log_probs[step] = log_prob     
                state = next_state
                global_step += self.nr_envs

                episode_infos = self.env.get_episode_infos(info)
                episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_info_buffer])
            
            acting_end_time = time.time()


            # Calculating advantages and returns
            with torch.no_grad():
                next_values = self.critic.get_value(batch.next_states).reshape(-1, 1)
            advantages, returns = calculate_gae_advantages_and_returns(batch.rewards, batch.dones, batch.values, next_values, self.gamma, self.gae_lambda)
            
            calc_adv_return_end_time = time.time()


            # Optimizing
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                for param_group in self.policy_optimizer.param_groups:
                    param_group["lr"] = learning_rate
                for param_group in self.critic_optimizer.param_groups:
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
            ratio_deltas = []

            for epoch in range(self.max_epochs):
                minibatch_indices = rng.choice(self.batch_size, size=self.minibatch_size, replace=False)

                ratio, loss1, pg_loss, entropy_loss, approx_kl_div = self.policy.loss(batch_states[minibatch_indices], batch_actions[minibatch_indices],
                                                                                                    batch_log_probs[minibatch_indices], batch_advantages[minibatch_indices])
                loss2, v_loss = self.critic.loss(batch_states[minibatch_indices], batch_returns[minibatch_indices])

                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss1.backward()
                loss2.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.critic_optimizer.step()

                episode_ratio_delta = self.delta_calc_operator(torch.abs(ratio - 1))

                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                loss_losses.append(loss1.item() + loss2.item())
                ratio_deltas.append(episode_ratio_delta.item())

                if episode_ratio_delta > self.max_ratio_delta:
                    break
            
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
            if self.track_console:
                rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
                self.log_console("global_step", global_step)
            else:
                rlx_logger.info(f"Step: {global_step}")

            if len(episode_info_buffer) > 0:
                self.log("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in episode_info_buffer]), global_step)
                self.log("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                names = list(episode_info_buffer[0].keys())
                for name in names:
                    if name != "r" and name != "l" and name != "t":
                        self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
            self.log("time/fps", int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time)), global_step)
            self.log("time/acting_time", acting_end_time - start_time, global_step)
            self.log("time/calc_adv_and_return_time", calc_adv_return_end_time - acting_end_time, global_step)
            self.log("time/optimizing_time", optimizing_end_time - calc_adv_return_end_time, global_step)
            self.log("time/saving_time", saving_end_time - optimizing_end_time, global_step)
            self.log("train/learning_rate", learning_rate, global_step)
            self.log("train/epochs", epoch + 1, global_step)
            self.log("train/ratio_delta", np.mean(ratio_deltas), global_step)
            self.log("train/last_approx_kl", approx_kl_div, global_step)
            self.log("train/policy_gradient_loss", np.mean(pg_losses), global_step)
            self.log("train/value_loss", np.mean(value_losses), global_step)
            self.log("train/entropy_loss", np.mean(entropy_losses), global_step)
            self.log("train/loss", np.mean(loss_losses), global_step)
            self.log("train/std", np.mean(np.exp(self.policy.policy_logstd.data.cpu().numpy())), global_step)
            self.log("train/explained_variance", explained_var, global_step)

            if self.track_console:
                rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │")


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()


    def save(self):
        file_path = self.save_path + "/model_best.pt"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, run_path, writer):
        checkpoint = torch.load(config.runner.load_model)
        config.algorithm = checkpoint["config_algorithm"]
        model = ESPO(config, env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        return model


    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, done, info = self.env.step(processed_action.cpu().numpy())
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")


@torch.jit.script
def calculate_gae_advantages_and_returns(rewards, dones, values, next_values, gamma: float, gae_lambda: float):
    delta = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    for t in range(values.shape[0] - 2, -1, -1):
        lastgaelam = advantages[t] = delta[t] + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
    returns = advantages + values
    return advantages, returns
