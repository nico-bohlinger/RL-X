import math
import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import wandb

from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.algorithms.flashsac.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.flashsac.pytorch.policy import get_policy
from rl_x.algorithms.flashsac.pytorch.critic import get_critic
from rl_x.algorithms.flashsac.pytorch.entropy_coefficient import get_entropy_coefficient
from rl_x.algorithms.flashsac.pytorch.layers import normalize_module
from rl_x.algorithms.flashsac.pytorch.reward_normalizer import RewardNormalizer
from rl_x.algorithms.flashsac.pytorch.noise_repeat import NoiseRepeat
from rl_x.algorithms.flashsac.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class FlashSAC:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.use_amp = config.algorithm.use_amp
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate_init = config.algorithm.learning_rate_init
        self.learning_rate_peak = config.algorithm.learning_rate_peak
        self.learning_rate_end = config.algorithm.learning_rate_end
        self.learning_rate_warmup_steps = config.algorithm.learning_rate_warmup_steps
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.policy_delay = config.algorithm.policy_delay
        self.gamma = config.algorithm.gamma
        self.n_steps = config.algorithm.n_steps
        self.tau = config.algorithm.tau
        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.normalized_g_max = config.algorithm.normalized_g_max
        self.normalize_reward = config.algorithm.normalize_reward
        self.noise_zeta_mu = config.algorithm.noise_zeta_mu
        self.noise_zeta_max = config.algorithm.noise_zeta_max
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.os_shape = self.train_env.single_observation_space.shape
        self.action_dim = self.train_env.single_action_space.shape[0]

        target_sigma = config.algorithm.target_entropy_sigma
        self.target_entropy = 0.5 * self.action_dim * math.log(2.0 * math.pi * math.e * target_sigma ** 2)

        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")

        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif config.algorithm.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)
        rlx_logger.info(f"Using device: {self.device}")

        if self.use_amp and self.device.type != "cuda":
            raise ValueError("AMP is only supported on CUDA devices.")

        torch.manual_seed(self.seed)

        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        self.target_critic = get_critic(config, self.train_env, self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.entropy_coefficient = get_entropy_coefficient(config, self.device)

        normalize_module(self.policy)
        normalize_module(self.critic)
        normalize_module(self.target_critic)

        fused = self.device.type == "cuda"
        total_updates = max(1, (self.total_timesteps // self.nr_envs) * self.updates_per_step)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate_peak, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_peak, fused=fused)
        self.entropy_coefficient_optimizer = optim.Adam(self.entropy_coefficient.parameters(), lr=self.learning_rate_peak, fused=fused)
        self.policy_scheduler = self.make_scheduler(self.policy_optimizer, total_updates)
        self.critic_scheduler = self.make_scheduler(self.critic_optimizer, total_updates)
        self.entropy_coefficient_scheduler = self.make_scheduler(self.entropy_coefficient_optimizer, total_updates)

        self.is_torch_data_interface = self.train_env.general_properties.data_interface_type == DataInterfaceType.TORCH

        self.noise_repeat = NoiseRepeat(self.nr_envs, self.action_dim, self.noise_zeta_mu, self.noise_zeta_max, self.device)
        self.reward_normalizer = RewardNormalizer(self.nr_envs, self.gamma, self.normalized_g_max, self.device)
        self.grad_scaler = GradScaler(device=self.device.type, enabled=self.use_amp)
        self.update_step_count = 0

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def make_scheduler(self, optimizer, total_updates):
        warmup = self.learning_rate_warmup_steps
        decay = max(1, total_updates - warmup)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                frac = step / warmup
                return (self.learning_rate_init + frac * (self.learning_rate_peak - self.learning_rate_init)) / self.learning_rate_peak
            progress = (step - warmup) / decay
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = self.learning_rate_end + (self.learning_rate_peak - self.learning_rate_end) * cosine
            return lr / self.learning_rate_peak
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    def update_step(self, states, next_states, actions, rewards, dones, truncations, effective_n_steps, do_policy):
        batch_size = states.shape[0]
        policy_loss_value = np.nan
        entropy_loss_value = np.nan
        entropy_value = np.nan
        policy_q_mean = np.nan
        alpha_value = np.nan

        if do_policy:
            self.policy.train()
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                all_observations = torch.cat([states, next_states], dim=0)
                all_means, all_stds = self.policy(all_observations)
                mean = all_means[:batch_size]
                std = all_stds[:batch_size]
                action, log_prob = self.policy.sample_and_log_prob(mean, std)
                for parameter in self.critic.parameters():
                    parameter.requires_grad = False
                self.critic.eval()
                q_values, _ = self.critic(states, action)
                q = torch.minimum(q_values[0], q_values[1])
                alpha = self.entropy_coefficient().detach()
                policy_loss = (alpha * log_prob - q).mean()
                entropy = -log_prob.mean()
            self.policy.eval()

            self.policy_optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.grad_scaler.scale(policy_loss).backward()
                self.grad_scaler.step(self.policy_optimizer)
                self.grad_scaler.update()
            else:
                policy_loss.backward()
                self.policy_optimizer.step()
            self.critic.train()
            for parameter in self.critic.parameters():
                parameter.requires_grad = True
            normalize_module(self.policy)
            self.policy_scheduler.step()

            entropy_loss = self.entropy_coefficient() * (entropy.detach().float() - self.target_entropy)
            self.entropy_coefficient_optimizer.zero_grad(set_to_none=True)
            entropy_loss.backward()
            self.entropy_coefficient_optimizer.step()
            self.entropy_coefficient_scheduler.step()

            policy_loss_value = policy_loss.item()
            entropy_loss_value = entropy_loss.item()
            entropy_value = entropy.item()
            alpha_value = alpha.item()
            policy_q_mean = q.mean().item()

        self.policy.eval()
        self.critic.train()
        self.target_critic.train()
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
            with torch.no_grad():
                next_mean, next_std = self.policy(next_states)
                next_action, next_log_prob = self.policy.sample_and_log_prob(next_mean, next_std)
                alpha = self.entropy_coefficient().detach()
                all_observations = torch.cat([states, next_states], dim=0)
                all_actions = torch.cat([actions, next_action], dim=0)
                _, target_log_probabilities = self.target_critic(all_observations, all_actions)
                next_log_probabilities = target_log_probabilities[:, batch_size:, :]
                bin_values = torch.linspace(self.v_min, self.v_max, self.nr_atoms, device=self.device, dtype=torch.float32)
                next_values = torch.sum(torch.exp(next_log_probabilities) * bin_values, dim=-1)
                minimum_value_indices = next_values.argmin(dim=0)
                selected_next_log_probabilities = torch.gather(
                    next_log_probabilities, 0, minimum_value_indices[None, :, None].expand(1, -1, self.nr_atoms),
                )[0]

                discount = (self.gamma ** effective_n_steps) * (1.0 - dones * (1.0 - truncations))
                target_bin_values = rewards.unsqueeze(-1) + discount.unsqueeze(-1) * (bin_values.unsqueeze(0) - (alpha * next_log_prob).unsqueeze(-1))
                target_bin_values = torch.clamp(target_bin_values, self.v_min, self.v_max)
                bin_width = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                target_bin_indices = (target_bin_values - self.v_min) / bin_width
                lower_bin_indices = torch.floor(target_bin_indices).long()
                upper_bin_indices = torch.clamp(lower_bin_indices + 1, 0, self.nr_atoms - 1)
                upper_bin_weights = target_bin_indices - lower_bin_indices.float()
                next_probabilities = torch.exp(selected_next_log_probabilities)
                lower_bin_weights = next_probabilities * (1.0 - upper_bin_weights)
                upper_bin_weights = next_probabilities * upper_bin_weights
                target_probabilities = torch.zeros((rewards.shape[0], self.nr_atoms), dtype=torch.float32, device=self.device)
                target_probabilities.scatter_add_(1, lower_bin_indices, lower_bin_weights)
                target_probabilities.scatter_add_(1, upper_bin_indices, upper_bin_weights)

            _, predicted_log_probabilities = self.critic(all_observations, all_actions)
            predicted_log_probabilities = predicted_log_probabilities[:, :batch_size, :]
            cross_entropy = -torch.sum(target_probabilities.unsqueeze(0) * predicted_log_probabilities, dim=-1)
            critic_loss = cross_entropy.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.grad_scaler.scale(critic_loss).backward()
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            critic_loss.backward()
            self.critic_optimizer.step()
        normalize_module(self.critic)
        with torch.no_grad():
            for critic_parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_parameter.data.mul_(1.0 - self.tau).add_(critic_parameter.data, alpha=self.tau)
        self.critic_scheduler.step()

        return {
            "loss/policy_loss": policy_loss_value,
            "loss/entropy_loss": entropy_loss_value,
            "loss/critic_loss": critic_loss.item(),
            "entropy/entropy": entropy_value,
            "entropy/alpha": alpha_value,
            "q_value/policy_q_mean": policy_q_mean,
            "q_value/target_q_mean": next_values.mean().item(),
            "lr/policy_learning_rate": self.policy_scheduler.get_last_lr()[0],
            "lr/critic_learning_rate": self.critic_scheduler.get_last_lr()[0],
        }


    def train(self):
        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.os_shape, self.action_dim, self.n_steps, self.gamma, self.device)
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        if not self.is_torch_data_interface:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Acting
            with torch.inference_mode():
                if global_step < self.learning_starts:
                    action = torch.empty((self.nr_envs, self.action_dim), dtype=torch.float32, device=self.device).uniform_(-1.0, 1.0)
                else:
                    with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                        mean, std = self.policy(state)
                        noise = self.noise_repeat.step()
                        action = self.policy.sample_with_noise(mean, std, noise, 1.0)
                processed_action = self.policy.get_processed_action(action)
                action_to_env = processed_action if self.is_torch_data_interface else processed_action.cpu().numpy()

            next_state, reward, terminated, truncated, info = self.train_env.step(action_to_env)
            done = terminated | truncated
            dones_this_rollout = 0
            if not self.is_torch_data_interface:
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                actual_next_state = next_state.clone()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = torch.tensor(
                            np.array(self.train_env.get_final_observation_at_index(info, i), dtype=np.float32),
                            dtype=torch.float32, device=self.device,
                        )
                        saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
                terminated_t = torch.tensor(terminated, dtype=torch.bool, device=self.device)
                truncated_t = torch.tensor(truncated, dtype=torch.bool, device=self.device)
                done_t = torch.tensor(done, dtype=torch.float32, device=self.device)
                truncated_f = truncated_t.float()
            else:
                dones_this_rollout += done.sum().item()
                reward_t = reward.float()
                terminated_t = terminated.bool()
                truncated_t = truncated.bool()
                done_t = done.float()
                actual_next_state = next_state
                has_actual_next_state = False
                if dones_this_rollout > 0:
                    environment_state = getattr(self.train_env, "env_state", None)
                    if environment_state is not None and hasattr(environment_state, "actual_next_observation"):
                        actual_next_observation = environment_state.actual_next_observation
                        actual_next_state = actual_next_observation if isinstance(actual_next_observation, torch.Tensor) else torch.from_dlpack(actual_next_observation)
                        has_actual_next_state = True
                    elif "final_observation" in info and isinstance(info["final_observation"], torch.Tensor):
                        actual_next_state = next_state.clone()
                        final_observation_mask = info.get("_final_observation", done).bool()
                        actual_next_state[final_observation_mask] = info["final_observation"][final_observation_mask]
                        has_actual_next_state = True
                truncated_f = truncated_t.float() if has_actual_next_state or dones_this_rollout == 0 else torch.zeros_like(done_t)
            logging_info = self.train_env.get_logging_info_dict(info)
            if self.is_torch_data_interface and dones_this_rollout > 0:
                episode_returns = logging_info.get("episode_return", logging_info.get("rollout/episode_return"))
                if episode_returns is not None:
                    if isinstance(episode_returns, torch.Tensor):
                        episode_returns = episode_returns.detach().cpu().numpy()
                    done_indices = done.detach().cpu().numpy().astype(bool)
                    episode_returns = np.asarray(episode_returns).reshape(-1)
                    if episode_returns.size == self.nr_envs:
                        episode_returns = episode_returns[done_indices]
                    if episode_returns.size == dones_this_rollout:
                        saving_return_buffer.extend(episode_returns.tolist())
            for k, info_value in logging_info.items():
                if isinstance(info_value, torch.Tensor):
                    info_value = info_value.detach().cpu().numpy()
                step_info_collection.setdefault(k, []).extend(np.asarray(info_value).reshape(-1).tolist())
            nr_episodes += dones_this_rollout

            if self.normalize_reward:
                self.reward_normalizer.update(reward_t, terminated_t, truncated_t)

            replay_buffer.add(state, actual_next_state, action, reward_t, done_t, truncated_f)
            state = next_state
            global_step += self.nr_envs

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_optimize = global_step >= self.learning_starts and replay_buffer.can_sample()
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_optimize and self.save_model and dones_this_rollout > 0 and bool(saving_return_buffer)
            should_log = global_step % self.logging_frequency == 0


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    states_b, next_states_b, actions_b, rewards_b, dones_b, truncs_b, eff_n_b = replay_buffer.sample(self.batch_size)
                    if self.normalize_reward:
                        rewards_b = self.reward_normalizer.normalize(rewards_b)
                    do_policy = self.update_step_count % self.policy_delay == 0
                    optimization_metrics = self.update_step(states_b, next_states_b, actions_b, rewards_b, dones_b, truncs_b, eff_n_b, do_policy)
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    self.update_step_count += 1
                    nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                with torch.inference_mode():
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    if not self.is_torch_data_interface:
                        eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                    eval_nr_episodes = 0
                    eval_episode_returns = np.zeros(self.nr_envs, dtype=np.float64)
                    eval_episode_lengths = np.zeros(self.nr_envs, dtype=np.int64)
                    while True:
                        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                            eval_mean, _ = self.policy(eval_state)
                            eval_action = self.policy.deterministic_action(eval_mean)
                            eval_action = self.policy.get_processed_action(eval_action)
                        eval_action_to_env = eval_action if self.is_torch_data_interface else eval_action.cpu().numpy()
                        eval_state, eval_reward, eval_term, eval_trunc, _ = self.eval_env.step(eval_action_to_env)
                        if not self.is_torch_data_interface:
                            eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                            eval_done = eval_term | eval_trunc
                        else:
                            eval_reward = eval_reward.detach().cpu().numpy()
                            eval_done = (eval_term | eval_trunc).detach().cpu().numpy()
                        eval_episode_returns += np.asarray(eval_reward)
                        eval_episode_lengths += 1
                        for i, single_done in enumerate(eval_done):
                            if single_done:
                                eval_nr_episodes += 1
                                evaluation_metrics_collection.setdefault("eval/episode_return", []).append(eval_episode_returns[i])
                                evaluation_metrics_collection.setdefault("eval/episode_length", []).append(eval_episode_lengths[i])
                                eval_episode_returns[i] = 0.0
                                eval_episode_lengths[i] = 0
                                if eval_nr_episodes == self.evaluation_episodes:
                                    break
                        if eval_nr_episodes == self.evaluation_episodes:
                            break
                    self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = float(np.mean(saving_return_buffer))
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)


            # Logging
            if should_log:
                self.start_logging(global_step)
                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes
                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    for info_name, info_vals in step_info_collection.items():
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(info_vals)
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {
                    key: float(np.nanmean(value))
                    for key, value in optimization_metrics_collection.items()
                    if not np.isnan(np.asarray(value)).all()
                }
                evaluation_metrics = {key: float(np.mean(value)) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(key, value, global_step)
                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}
                self.end_logging()

            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)


    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self, wandb_commit=True):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache, commit=wandb_commit)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        file_path = self.save_path + "/best.model"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "entropy_coefficient_state_dict": self.entropy_coefficient.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "entropy_coefficient_optimizer_state_dict": self.entropy_coefficient_optimizer.state_dict(),
            "policy_scheduler_state_dict": self.policy_scheduler.state_dict(),
            "critic_scheduler_state_dict": self.critic_scheduler.state_dict(),
            "entropy_coefficient_scheduler_state_dict": self.entropy_coefficient_scheduler.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict(),
            "reward_normalizer_state_dict": self.reward_normalizer.state_dict(),
            "update_step_count": self.update_step_count,
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif config.algorithm.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"
        checkpoint = torch.load(config.runner.load_model, map_location=device_name, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FlashSAC(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        model.entropy_coefficient.load_state_dict(checkpoint["entropy_coefficient_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.entropy_coefficient_optimizer.load_state_dict(checkpoint["entropy_coefficient_optimizer_state_dict"])
        model.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
        model.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        model.entropy_coefficient_scheduler.load_state_dict(checkpoint["entropy_coefficient_scheduler_state_dict"])
        model.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        model.reward_normalizer.load_state_dict(checkpoint["reward_normalizer_state_dict"])
        model.update_step_count = checkpoint["update_step_count"]
        return model


    def test(self, episodes):
        with torch.inference_mode():
            self.set_eval_mode()
            state, _ = self.eval_env.reset()
            episode_returns = np.zeros(self.nr_envs, dtype=np.float64)
            completed_episodes = 0
            while completed_episodes < episodes:
                if not self.is_torch_data_interface:
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    mean, _ = self.policy(state)
                    action = self.policy.get_processed_action(self.policy.deterministic_action(mean))
                action_to_env = action if self.is_torch_data_interface else action.cpu().numpy()
                state, reward, terminated, truncated, _ = self.eval_env.step(action_to_env)
                if isinstance(reward, torch.Tensor):
                    reward = reward.detach().cpu().numpy()
                    done = (terminated | truncated).detach().cpu().numpy()
                else:
                    done = terminated | truncated
                episode_returns += np.asarray(reward)
                for index, single_done in enumerate(done):
                    if single_done:
                        completed_episodes += 1
                        rlx_logger.info(f"Episode {completed_episodes} - Return: {episode_returns[index]}")
                        episode_returns[index] = 0.0
                        if completed_episodes == episodes:
                            break


    def set_train_mode(self):
        self.policy.eval()
        self.critic.train()
        self.target_critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()
        self.target_critic.eval()


    def general_properties():
        return GeneralProperties
