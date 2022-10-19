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

from rl_x.algorithms.espo.torchscript.agent import get_agent
from rl_x.algorithms.espo.torchscript.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class ESPO:
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
        self.batch_size = config.algorithm.nr_envs * config.algorithm.nr_steps

        if config.algorithm.delta_calc_operator == "mean":
            self.delta_calc_operator = torch.mean
        elif config.algorithm.delta_calc_operator == "median":
            self.delta_calc_operator = torch.median
        else:
            raise ValueError("Unknown delta_calc_operator")

        self.device = torch.device("cuda" if config.algorithm.device == "cuda" and torch.cuda.is_available() else "cpu")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        self.agent = get_agent(config, env).to(self.device)
        self.agent = torch.jit.script(self.agent)
        self.agent_optimizer = optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

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

        saving_return_buffer = deque(maxlen=100)
        rng = default_rng()
        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
        

            # Acting
            state = torch.tensor(self.env.reset(), dtype=torch.float32).to(self.device)
            episode_info_buffer = deque(maxlen=100)
            for step in range(self.nr_steps):
                with torch.no_grad():
                    action, log_prob = self.agent.get_action_logprob(state)
                    value = self.agent.get_value(state)
                next_state, reward, done, info = self.env.step(action.cpu().numpy())
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

                episode_info_buffer.extend(self.env.get_episode_infos(info))
                if len(episode_info_buffer) > 0:
                    ep_info_returns = [ep_info["r"] for ep_info in episode_info_buffer]
                    saving_return_buffer.extend(ep_info_returns)
            
            acting_end_time = time.time()


            # Calculating advantages and returns
            with torch.no_grad():
                next_values = self.agent.get_value(batch.next_states).squeeze()
            advantages, returns = calculate_gae_advantages_and_returns(batch.rewards, batch.dones, batch.values, next_values, self.gamma, self.gae_lambda)
            
            calc_adv_return_end_time = time.time()


            # Optimizing
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                for param_group in self.agent_optimizer.param_groups:
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

                ratio, loss, pg_loss, v_loss, entropy_loss, approx_kl_div = self.agent.loss(batch_states[minibatch_indices], batch_actions[minibatch_indices],
                                                                                            batch_log_probs[minibatch_indices], batch_returns[minibatch_indices],
                                                                                            batch_advantages[minibatch_indices])

                self.agent_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.agent_optimizer.step()

                episode_ratio_delta = self.delta_calc_operator(torch.abs(ratio - 1))

                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                loss_losses.append(loss.item())
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
                self.writer.add_scalar("train/epochs", epoch + 1, global_step)
                self.writer.add_scalar("train/ratio_delta", np.mean(ratio_deltas), global_step)
                self.writer.add_scalar("train/last_approx_kl", approx_kl_div, global_step)
                self.writer.add_scalar("train/policy_gradient_loss", np.mean(pg_losses), global_step)
                self.writer.add_scalar("train/value_loss", np.mean(value_losses), global_step)
                self.writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), global_step)
                self.writer.add_scalar("train/loss", np.mean(loss_losses), global_step)
                self.writer.add_scalar("train/std", np.mean(np.exp(self.agent.actor_logstd.data.cpu().numpy())), global_step)
                self.writer.add_scalar("train/explained_variance", explained_var, global_step)

            rlx_logger.info(f"Step: {global_step}")


    def set_train_mode(self):
        self.agent.train()


    def set_eval_mode(self):
        self.agent.eval()


    def save(self):
        file_path = self.save_path + "/model_best.pt"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "agent_state_dict": self.agent.state_dict(),
            "agent_optimizer_state_dict": self.agent_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, writer):
        checkpoint = torch.load(config.runner.load_model)
        config.algorithm = checkpoint["config_algorithm"]
        model = ESPO(config, env, writer)
        model.agent.load_state_dict(checkpoint["agent_state_dict"])
        model.agent_optimizer.load_state_dict(checkpoint["agent_optimizer_state_dict"])

        return model


    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                action, _ = self.agent.get_action_logprob(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, done, info = self.env.step(action.cpu().numpy())
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
