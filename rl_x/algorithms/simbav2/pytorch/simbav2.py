import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.simbav2.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.simbav2.pytorch.policy import get_policy
from rl_x.algorithms.simbav2.pytorch.critic import get_critic
from rl_x.algorithms.simbav2.pytorch.entropy_coefficient import get_entropy_coefficient
from rl_x.algorithms.simbav2.pytorch.layers import l2normalize_parameters
from rl_x.algorithms.simbav2.pytorch.replay_buffer import ReplayBuffer
from rl_x.algorithms.simbav2.pytorch.normalizer import update_normalizer, update_reward_normalizer

rlx_logger = logging.getLogger("rl_x")


class SimbaV2:
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
        self.compile_mode = config.algorithm.compile_mode
        self.bf16_mixed_precision_training = config.algorithm.bf16_mixed_precision_training
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate_init = config.algorithm.learning_rate_init
        self.learning_rate_end = config.algorithm.learning_rate_end
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.gamma = config.algorithm.gamma
        self.tau = config.algorithm.tau
        self.use_cdq = config.algorithm.use_cdq
        self.nr_bins = config.algorithm.nr_bins
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.normalize_observation = config.algorithm.normalize_observation
        self.normalize_reward = config.algorithm.normalize_reward
        self.normalized_g_max = config.algorithm.normalized_g_max
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif config.algorithm.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)
        rlx_logger.info(f"Using device: {self.device}")
        rlx_logger.info(f"Using discount factor: {self.gamma}")

        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training is only supported on CUDA devices.")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        self.target_critic = get_critic(config, self.train_env, self.device)
        l2normalize_parameters(self.policy)
        l2normalize_parameters(self.critic)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.entropy_coefficient = get_entropy_coefficient(config, self.train_env, self.device)

        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate_init, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_init, fused=fused)
        self.entropy_optimizer = optim.Adam([self.entropy_coefficient.log_alpha], lr=self.learning_rate_init, fused=fused)

        total_updates = max(1, (self.total_timesteps // self.nr_envs) * self.updates_per_step)
        self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=self.learning_rate_end / self.learning_rate_init, total_iters=total_updates)
        self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=self.learning_rate_end / self.learning_rate_init, total_iters=total_updates)
        self.entropy_scheduler = optim.lr_scheduler.LinearLR(self.entropy_optimizer, start_factor=1.0, end_factor=self.learning_rate_end / self.learning_rate_init, total_iters=total_updates)

        self.observation_normalizer_state = {
            "mean": np.zeros(self.train_env.single_observation_space.shape, dtype=np.float32),
            "var": np.ones(self.train_env.single_observation_space.shape, dtype=np.float32),
            "count": np.float32(1e-4),
        }
        self.reward_normalizer_state = {
            "G_r": np.zeros((self.nr_envs,), dtype=np.float32),
            "G_r_max": np.float32(0.0),
            "rms_mean": np.float32(0.0),
            "rms_var": np.float32(1.0),
            "rms_count": np.float32(1e-4),
        }

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        @torch.compile(mode=self.compile_mode)
        def policy_and_entropy_loss_fn(states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                actions, _, log_probs = self.policy.get_action(states)
                q_values, _ = self.critic(states, actions)
                q = torch.min(q_values, dim=0).values if self.use_cdq else q_values
                alpha = self.entropy_coefficient()
                alpha_detach = alpha.detach()
                policy_loss = (alpha_detach * log_probs - q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float("inf"))
            self.policy_optimizer.step()

            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                entropy = -log_probs.detach().mean()
                entropy_loss = self.entropy_coefficient.loss(entropy)

            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            entropy_grad_norm = self.entropy_coefficient.log_alpha.grad.detach().norm(2)
            self.entropy_optimizer.step()
            return policy_loss, entropy_loss, q.mean(), entropy, alpha_detach, policy_grad_norm, entropy_grad_norm


        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, terminations):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_values, next_log_probabilities = self.target_critic(next_states, next_actions)
                    if self.use_cdq:
                        minimum_value_indices = torch.argmin(next_values, dim=0)
                        selected_next_log_probabilities = next_log_probabilities.permute(1, 0, 2)[torch.arange(states.shape[0], device=self.device), minimum_value_indices]
                    else:
                        selected_next_log_probabilities = next_log_probabilities
                    bin_values = torch.linspace(self.v_min, self.v_max, self.nr_bins, dtype=torch.float32, device=self.device)
                    target_bin_values = rewards[:, None] + self.gamma * (1.0 - terminations[:, None]) * (bin_values[None, :] - self.entropy_coefficient() * next_log_probs[:, None])
                    target_bin_values = torch.clamp(target_bin_values, self.v_min, self.v_max)
                    target_bin_indices = (target_bin_values - self.v_min) / ((self.v_max - self.v_min) / (self.nr_bins - 1))
                    lower_bin_indices = torch.floor(target_bin_indices).long()
                    upper_bin_indices = torch.clamp(lower_bin_indices + 1, 0, self.nr_bins - 1)
                    upper_bin_probabilities = target_bin_indices - lower_bin_indices.float()
                    next_probabilities = torch.exp(selected_next_log_probabilities).float()
                    target_probabilities = torch.zeros((states.shape[0], self.nr_bins), dtype=torch.float32, device=self.device)
                    target_probabilities.scatter_add_(1, lower_bin_indices, next_probabilities * (1.0 - upper_bin_probabilities))
                    target_probabilities.scatter_add_(1, upper_bin_indices, next_probabilities * upper_bin_probabilities)
                    target_q = (target_probabilities * bin_values[None]).sum(dim=-1)
                predicted_q_values, predicted_log_probabilities = self.critic(states, actions)
                if self.use_cdq:
                    cross_entropy = -(target_probabilities[None] * predicted_log_probabilities).sum(dim=-1)
                    critic_loss = cross_entropy.mean(dim=1).sum()
                    q1_mean = predicted_q_values[0].mean()
                    q2_mean = predicted_q_values[1].mean()
                else:
                    critic_loss = -(target_probabilities * predicted_log_probabilities).sum(dim=-1).mean()
                    q1_mean = predicted_q_values.mean()
                    q2_mean = q1_mean

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float("inf"))
            self.critic_optimizer.step()
            return critic_loss, target_q.mean(), q1_mean, q2_mean, critic_grad_norm


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.rng, self.device)
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
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
            torch.compiler.cudagraph_mark_step_begin()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Acting
            dones_this_rollout = 0
            if self.normalize_observation:
                self.observation_normalizer_state = update_normalizer(self.observation_normalizer_state, state)
                processed_state = (state - self.observation_normalizer_state["mean"]) / np.sqrt(self.observation_normalizer_state["var"] + 1e-8)
            else:
                processed_state = state
            if global_step < self.learning_starts:
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    action, processed_action, _ = self.policy.get_action(torch.tensor(processed_state, dtype=torch.float32, device=self.device))
                action = action.cpu().numpy()
                processed_action = processed_action.cpu().numpy()

            next_state, reward, terminated, truncated, info = self.train_env.step(processed_action)
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            if self.normalize_reward:
                self.reward_normalizer_state = update_reward_normalizer(self.reward_normalizer_state, reward, terminated, truncated, self.gamma)
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)
                    if self.normalize_observation:
                        observation_mean = torch.tensor(self.observation_normalizer_state["mean"], dtype=torch.float32, device=self.device)
                        observation_std = torch.tensor(np.sqrt(self.observation_normalizer_state["var"] + 1e-8), dtype=torch.float32, device=self.device)
                        batch_states = (batch_states - observation_mean) / observation_std
                        batch_next_states = (batch_next_states - observation_mean) / observation_std
                    if self.normalize_reward:
                        reward_denom = max(np.sqrt(self.reward_normalizer_state["rms_var"] + 1e-8), self.reward_normalizer_state["G_r_max"] / self.normalized_g_max)
                        batch_rewards = batch_rewards / reward_denom
                    policy_loss, entropy_loss, policy_q_mean, entropy, alpha, policy_grad_norm, entropy_grad_norm = policy_and_entropy_loss_fn(batch_states)
                    l2normalize_parameters(self.policy)
                    critic_loss, target_q_mean, q1_mean, q2_mean, critic_grad_norm = critic_loss_fn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations)
                    l2normalize_parameters(self.critic)
                    with torch.no_grad():
                        for parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
                            target_parameter.data.mul_(1.0 - self.tau).add_(parameter.data, alpha=self.tau)
                    self.policy_scheduler.step()
                    self.critic_scheduler.step()
                    self.entropy_scheduler.step()

                    optimization_metrics = {
                        "entropy/alpha": alpha.item(),
                        "entropy/entropy": entropy.item(),
                        "gradients/policy_grad_norm": policy_grad_norm.item(),
                        "gradients/critic_grad_norm": critic_grad_norm.item(),
                        "gradients/entropy_grad_norm": entropy_grad_norm.item(),
                        "loss/critic_loss": critic_loss.item(),
                        "loss/policy_loss": policy_loss.item(),
                        "loss/entropy_loss": entropy_loss.item(),
                        "lr/policy_learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                        "lr/critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
                        "q_value/target_q_mean": target_q_mean.item(),
                        "q_value/q1_mean": q1_mean.item(),
                        "q_value/q2_mean": q2_mean.item(),
                        "q_value/policy_q_mean": policy_q_mean.item(),
                        "reward/mean": batch_rewards.mean().item(),
                    }
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    torch.compiler.cudagraph_mark_step_begin()
                    processed_eval_state = (eval_state - self.observation_normalizer_state["mean"]) / np.sqrt(self.observation_normalizer_state["var"] + 1e-8) if self.normalize_observation else eval_state
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        eval_processed_action = self.policy.get_deterministic_action(torch.tensor(processed_eval_state, dtype=torch.float32, device=self.device))
                    eval_state, _, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.cpu().numpy())
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
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
                    for info_name, info_values in step_info_collection.items():
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(info_values)
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value

                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
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
            "log_alpha": self.entropy_coefficient.log_alpha,
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
            "observation_normalizer_state": self.observation_normalizer_state,
            "reward_normalizer_state": self.reward_normalizer_state,
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = SimbaV2(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        with torch.no_grad():
            model.entropy_coefficient.log_alpha.copy_(checkpoint["log_alpha"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        model.observation_normalizer_state = checkpoint["observation_normalizer_state"]
        model.reward_normalizer_state = checkpoint["reward_normalizer_state"]
        return model


    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                processed_state = (state - self.observation_normalizer_state["mean"]) / np.sqrt(self.observation_normalizer_state["var"] + 1e-8) if self.normalize_observation else state
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    processed_action = self.policy.get_deterministic_action(torch.tensor(processed_state, dtype=torch.float32, device=self.device))
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()
        self.target_critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()
        self.target_critic.eval()


    def general_properties():
        return GeneralProperties
