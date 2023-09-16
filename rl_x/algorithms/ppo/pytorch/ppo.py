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

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.algorithms.ppo.pytorch.policy import get_policy
from rl_x.algorithms.ppo.pytorch.critic import get_critic
from rl_x.algorithms.ppo.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class PPO:
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
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_minibatches = self.batch_size // self.minibatch_size

        self.device = torch.device("cuda" if config.algorithm.device == "gpu" and torch.cuda.is_available() else "cpu")
        rlx_logger.info(f"Using device: {self.device}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape

        self.policy = get_policy(config, env, self.device).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

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
            terminations = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            log_probs = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            advantages = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            returns = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        episode_info_buffer = deque(maxlen=100 * self.nr_envs)
        time_metrics_buffer = deque(maxlen=1)
        optimization_metrics_buffer = deque(maxlen=1)
        
        state = torch.tensor(self.env.reset(), dtype=torch.float32).to(self.device)
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
        

            # Acting
            episode_info_buffer = deque(maxlen=100)
            for step in range(self.nr_steps):
                with torch.no_grad():
                    action, processed_action, log_prob = self.policy.get_action_logprob(state)
                    value = self.critic.get_value(state)
                next_state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                actual_next_state = next_state.clone()
                for i, single_done in enumerate(done):
                    if single_done:
                        maybe_final_observation = self.env.get_final_observation(info, i)
                        if maybe_final_observation is not None:
                            actual_next_state[i] = torch.tensor(np.array(maybe_final_observation), dtype=torch.float32).to(self.device)
                        nr_episodes += 1

                batch.states[step] = state
                batch.actions[step] = action
                batch.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                batch.values[step] = value.reshape(-1)
                batch.terminations[step]= torch.tensor(terminated, dtype=torch.float32).to(self.device)
                batch.log_probs[step] = log_prob     
                state = next_state
                global_step += self.nr_envs

                episode_infos = self.env.get_episode_infos(info)
                episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info])
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            with torch.no_grad():
                next_value = self.critic.get_value(actual_next_state).reshape(1, -1)
            lastgaelam = 0
            for t in reversed(range(self.nr_steps)):
                if t == self.nr_steps - 1:
                    nextvalues = next_value
                else:
                    nextvalues = batch.values[t + 1]
                not_terminated = 1.0 - batch.terminations[t]
                delta = batch.rewards[t] + self.gamma * nextvalues * not_terminated - batch.values[t]
                batch.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * not_terminated * lastgaelam
            batch.returns = batch.advantages + batch.values
            
            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                for param_group in self.policy_optimizer.param_groups + self.critic_optimizer.param_groups:
                    param_group["lr"] = learning_rate
            
            batch_states = batch.states.reshape((-1,) + self.os_shape)
            batch_actions = batch.actions.reshape((-1,) + self.as_shape)
            batch_advantages = batch.advantages.reshape(-1)
            batch_returns = batch.returns.reshape(-1)
            batch_values = batch.values.reshape(-1)
            batch_log_probs = batch.log_probs.reshape(-1)

            metrics_list = []
            batch_indices = np.arange(self.batch_size)
            for epoch in range(self.nr_epochs):
                approx_kl_divs = []
                np.random.shuffle(batch_indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    # Policy loss
                    new_log_prob, entropy = self.policy.get_logprob_entropy(batch_states[minibatch_indices], batch_actions[minibatch_indices])
                    new_value = self.critic.get_value(batch_states[minibatch_indices])
                    logratio = new_log_prob - batch_log_probs[minibatch_indices]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        log_ratio = new_log_prob - batch_log_probs[minibatch_indices]
                        approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div.item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())

                    minibatch_advantages = batch_advantages[minibatch_indices]
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

                    pg_loss1 = -minibatch_advantages * ratio
                    pg_loss2 = -minibatch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()

                    entropy_loss = entropy.mean()

                    # Critic loss
                    new_value = new_value.reshape(-1)
                    critic_loss = torch.mean(0.5 * (new_value - batch_returns[minibatch_indices]) ** 2)

                    # Combine losses
                    loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss 

                    # Backprop
                    self.policy_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()

                    policy_grad_norm = 0.0
                    critic_grad_norm = 0.0
                    for param in self.policy.parameters():
                        policy_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                    for param in self.critic.parameters():
                        critic_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                    policy_grad_norm = policy_grad_norm ** 0.5
                    critic_grad_norm = critic_grad_norm ** 0.5

                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()
                    self.critic_optimizer.step()

                    # Create metrics
                    metrics = {
                        "loss/policy_gradient_loss": pg_loss.item(),
                        "loss/critic_loss": critic_loss.item(),
                        "loss/entropy_loss": entropy_loss.item(),
                        "policy_ratio/clip_fraction": clip_fraction.item(),
                        "gradients/policy_grad_norm": policy_grad_norm,
                        "gradients/critic_grad_norm": critic_grad_norm,
                    }
                    metrics_list.append(metrics)

            y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            metrics = {key: np.mean([metrics[key] for metrics in metrics_list]) for key in metrics_list[0].keys()}
            metrics["lr/learning_rate"] = learning_rate
            metrics["v_value/explained_variance"] = explained_var
            metrics["policy_ratio/approx_kl"] = np.mean(approx_kl_divs)
            metrics["policy/std_dev"] = 0 if self.env.get_action_space_type() == ActionSpaceType.DISCRETE else np.mean(np.exp(self.policy.policy_logstd.data.cpu().numpy()))
            optimization_metrics_buffer.append(metrics)

            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Saving
            # Only save when the total return buffer (over multiple updates) isn't empty
            # Also only save when the episode info buffer isn't empty -> there were finished episodes this update
            if self.save_model and saving_return_buffer and episode_info_buffer:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - optimizing_end_time

            time_metrics["time/fps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))

            time_metrics_buffer.append(time_metrics)


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            if len(episode_info_buffer) > 0:
                self.log("rollout/episode_reward", np.mean([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info]), global_step)
                self.log("rollout/episode_length", np.mean([ep_info["l"] for ep_info in episode_info_buffer if "l" in ep_info]), global_step)
                names = list(episode_info_buffer[0].keys())
                for name in names:
                    if name != "r" and name != "l" and name != "t":
                        self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info]), global_step)
            mean_time_metrics = {key: np.mean([metrics[key] for metrics in time_metrics_buffer]) for key in sorted(time_metrics_buffer[0].keys())}
            mean_optimization_metrics = {key: np.mean([metrics[key] for metrics in optimization_metrics_buffer]) for key in sorted(optimization_metrics_buffer[0].keys())}
            combined_metrics = {**steps_metrics, **mean_time_metrics, **mean_optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            episode_info_buffer.clear()
            time_metrics_buffer.clear()
            optimization_metrics_buffer.clear()

            self.end_logging()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


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
        model = PPO(config, env, run_path, writer)
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
                state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
                done = terminated | truncated
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()

