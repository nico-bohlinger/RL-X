import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.mpo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.mpo.pytorch.policy import get_policy
from rl_x.algorithms.mpo.pytorch.critic import get_critic
from rl_x.algorithms.mpo.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0

class MPO:
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
        
        # Hyperparameters
        self.learning_rate = config.algorithm.learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.n_step = config.algorithm.n_step
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        
        # MPO Constraints
        self.epsilon_mean = config.algorithm.epsilon_mean
        self.epsilon_std = config.algorithm.epsilon_std
        self.epsilon_penalty = config.algorithm.epsilon_penalty
        self.epsilon_non_parametric = config.algorithm.epsilon_non_parametric
        self.action_sampling_number = int(config.algorithm.action_sampling_number)
        
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

        if self.bf16_mixed_precision_training and self.device.type != "cuda":
            raise ValueError("bfloat16 mixed precision training is only supported on CUDA devices.")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        self.action_dim = self.train_env.single_action_space.shape[0]

        # Components
        self.policy = get_policy(config, self.train_env, self.device)
        self.target_policy = get_policy(config, self.train_env, self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.critic = get_critic(config, self.train_env, self.device)
        
        # Dual Variables (Lagrange Multipliers)
        self.log_eta = torch.tensor([1.0], requires_grad=True, device=self.device)
        self.log_alpha_mean = torch.tensor([1.0] * self.action_dim, requires_grad=True, device=self.device)
        self.log_alpha_std = torch.tensor([5.0] * self.action_dim, requires_grad=True, device=self.device)
        self.log_penalty_temperature = torch.tensor([1.0], requires_grad=True, device=self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, fused=False) # fused=False safer for MPO complex loss
        self.q_optimizer = optim.Adam(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate, fused=True)
        self.dual_optimizer = optim.Adam([
            self.log_eta, 
            self.log_alpha_mean, 
            self.log_alpha_std, 
            self.log_penalty_temperature
        ], lr=self.dual_learning_rate)

        if self.anneal_learning_rate:
            total_iters = self.total_timesteps // self.nr_envs
            self.q_scheduler = optim.lr_scheduler.LinearLR(self.q_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)
            self.dual_scheduler = optim.lr_scheduler.LinearLR(self.dual_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    def train(self):
        @torch.compile(mode=self.compile_mode)
        def mpo_loss_fn(states):
            # MPO Policy Step (E-Step & M-Step combined in optimization)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                
                # --- 1. Sample actions from target policy (Non-parametric construction) ---
                with torch.no_grad():
                    # Shape: (B, action_dim)
                    target_mus, target_stds = self.target_policy(states) 
                    
                    target_dist_independent = torch.distributions.Independent(
                        torch.distributions.Normal(loc=target_mus, scale=target_stds), reinterpreted_batch_ndims=1
                    )
                    
                    # Sample K actions: (K, B, action_dim)
                    sampled_actions = target_dist_independent.sample(torch.Size([self.action_sampling_number]))
                    
                    # Expand states: (K, B, state_dim)
                    expanded_states = states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)
                    
                    # Flatten for Q-evaluation: (K*B, ...)
                    flat_states = expanded_states.reshape(-1, expanded_states.shape[-1])
                    flat_actions = sampled_actions.reshape(-1, sampled_actions.shape[-1])
                    
                    # Evaluate Q-values (Min Q)
                    q1_t = self.critic.q1_target(flat_states, flat_actions)
                    q2_t = self.critic.q2_target(flat_states, flat_actions)
                    q_values = torch.min(q1_t, q2_t).reshape(self.action_sampling_number, -1) # (K, B)

                # --- 2. Calculate Non-parametric weights (E-Step) ---
                eta = F.softplus(self.log_eta) + _MPO_FLOAT_EPSILON
                
                # Normalized weights: w_ij = exp(Q_ij / eta) / Z
                normalized_weights = F.softmax(q_values / eta, dim=0) # (K, B)
                
                # --- 3. Eta Loss (Dual Ascent) ---
                q_logsumexp = torch.logsumexp(q_values / eta, dim=0) # (B,)
                log_num_actions = torch.log(torch.tensor(self.action_sampling_number, dtype=torch.float32, device=self.device))
                
                loss_eta = self.epsilon_non_parametric + torch.mean(q_logsumexp) - log_num_actions
                loss_eta = eta * loss_eta
                
                # --- 4. Out-of-bound Penalty Loss ---
                penalty_temp = F.softplus(self.log_penalty_temperature) + _MPO_FLOAT_EPSILON
                diff_out_of_bound = sampled_actions - torch.clamp(sampled_actions, -1, 1) # (K, B, A)
                cost_out_of_bound = -torch.linalg.norm(diff_out_of_bound, dim=-1) # (K, B)
                
                penalty_logsumexp = torch.logsumexp(cost_out_of_bound / penalty_temp, dim=0)
                loss_penalty = self.epsilon_penalty + torch.mean(penalty_logsumexp) - log_num_actions
                loss_penalty = penalty_temp * loss_penalty
                
                # Add penalty influence to non-parametric distribution
                penalty_weights = F.softmax(cost_out_of_bound / penalty_temp.detach(), dim=0)
                combined_weights = normalized_weights + penalty_weights # Paper trick to combine
                loss_eta += loss_penalty

                # --- 5. Policy Regression (M-Step) ---
                # Get current policy distribution parameters
                online_mus, online_stds = self.policy(states) # (B, A)
                
                # Decoupled KL constraints (Mean)
                online_dist_mean = torch.distributions.Independent(
                    torch.distributions.Normal(loc=online_mus, scale=target_stds.detach()), reinterpreted_batch_ndims=1
                )
                
                # Log prob of the *target samples* under current mean
                # We need to broadcast online dist to shape (K, B, A) effectively
                # sample shape: (K, B, A). log_prob returns (K, B)
                log_prob_mean = online_dist_mean.log_prob(sampled_actions) 
                
                # Weighted Likelihood
                loss_policy_mean = -torch.sum(log_prob_mean * combined_weights.detach(), dim=0).mean()
                
                # KL Constraint for Mean
                kl_mean = torch.distributions.kl.kl_divergence(
                    target_dist_independent.base_dist, online_dist_mean.base_dist
                ) # (B, A) - per dimension
                mean_kl_mean = torch.mean(kl_mean, dim=0) # (A,)
                
                alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON
                loss_kl_mean = torch.sum(alpha_mean.detach() * mean_kl_mean)
                loss_alpha_mean = torch.sum(alpha_mean * (self.epsilon_mean - mean_kl_mean.detach()))

                # Decoupled KL constraints (Std)
                online_dist_std = torch.distributions.Independent(
                    torch.distributions.Normal(loc=target_mus.detach(), scale=online_stds), reinterpreted_batch_ndims=1
                )
                log_prob_std = online_dist_std.log_prob(sampled_actions)
                
                loss_policy_std = -torch.sum(log_prob_std * combined_weights.detach(), dim=0).mean()
                
                # KL Constraint for Std
                kl_std = torch.distributions.kl.kl_divergence(
                    target_dist_independent.base_dist, online_dist_std.base_dist
                )
                mean_kl_std = torch.mean(kl_std, dim=0)
                
                # Softplus on alpha_std (CleanRL does logaddexp, softplus is safer generally)
                alpha_std = F.softplus(self.log_alpha_std) + _MPO_FLOAT_EPSILON
                loss_kl_std = torch.sum(alpha_std.detach() * mean_kl_std)
                loss_alpha_std = torch.sum(alpha_std * (self.epsilon_std - mean_kl_std.detach()))

                # Total losses
                total_policy_loss = loss_policy_mean + loss_policy_std + loss_kl_mean + loss_kl_std
                total_dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta

            return total_policy_loss, total_dual_loss, q_values, mean_kl_mean, mean_kl_std, alpha_mean, alpha_std, eta


        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, dones, effective_n_steps):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    # TD(n) Target
                    # Sample action from target policy
                    next_actions_dist = self.target_policy.get_distribution(next_states)
                    next_actions = next_actions_dist.sample()
                    
                    # Target Q
                    next_q1 = self.critic.q1_target(next_states, next_actions)
                    next_q2 = self.critic.q2_target(next_states, next_actions)
                    min_next_q = torch.minimum(next_q1, next_q2)
                    
                    # N-step return
                    gammas = self.gamma ** effective_n_steps.reshape(-1, 1)
                    target_q = rewards.reshape(-1, 1) + gammas * (1 - dones.reshape(-1, 1)) * min_next_q

                q1 = self.critic.q1(states, actions)
                q2 = self.critic.q2(states, actions)
                
                q1_loss = F.mse_loss(q1, target_q)
                q2_loss = F.mse_loss(q2, target_q)
                q_loss = q1_loss + q2_loss
            
            return q_loss

        self.set_train_mode()

        # Initialize Replay Buffer (FastTD3 variant provided)
        replay_buffer = ReplayBuffer(
            int(self.buffer_size // self.nr_envs), 
            self.nr_envs, 
            self.train_env.single_observation_space.shape, 
            self.train_env.single_action_space.shape,
            self.n_step,
            self.gamma,
            self.device
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        
        # Metric collections
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

            # --- Acting ---
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                # Random action
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                # Inverse scale to [-1, 1] for storage
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    # Get action returns (raw_action, scaled_action, log_prob)
                    action_tensor, processed_action_tensor, _ = self.policy.get_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                action = action_tensor.cpu().numpy()
                processed_action = processed_action_tensor.cpu().numpy()
            
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
            
            # Add to buffer (FastTD3 signature: states, next_states, actions, rewards, dones, truncations)
            replay_buffer.add(
                torch.tensor(state, dtype=torch.float32, device=self.device),
                torch.tensor(actual_next_state, dtype=torch.float32, device=self.device),
                torch.tensor(action, dtype=torch.float32, device=self.device),
                torch.tensor(reward, dtype=torch.float32, device=self.device),
                torch.tensor(terminated, dtype=torch.float32, device=self.device),
                torch.tensor(truncated, dtype=torch.float32, device=self.device)
            )

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)

            # --- Logic ---
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0

            # --- Optimizing ---
            if should_optimize:
                # Sample
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncs, effective_n_steps = replay_buffer.sample(self.batch_size)

                # Critic Update
                q_loss = critic_loss_fn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, effective_n_steps)
                
                self.q_optimizer.zero_grad()
                q_loss.backward()
                # Clip Norm is common in MPO
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 40.0)
                self.q_optimizer.step()

                # Policy & Dual Update
                policy_loss, dual_loss, q_values, mean_kl_mean, mean_kl_std, alpha_mean, alpha_std, eta = mpo_loss_fn(batch_states)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 40.0)
                self.policy_optimizer.step()

                self.dual_optimizer.zero_grad()
                dual_loss.backward()
                # CleanRL implementation clips dual params manually after step
                self.dual_optimizer.step()
                
                with torch.no_grad():
                     self.log_eta.data.clamp_(min=_MIN_LOG_TEMPERATURE)
                     self.log_alpha_mean.data.clamp_(min=_MIN_LOG_ALPHA)
                     self.log_alpha_std.data.clamp_(min=_MIN_LOG_ALPHA)
                     self.log_penalty_temperature.data.clamp_(min=_MIN_LOG_TEMPERATURE)

                # Target Update
                with torch.no_grad():
                    for param, target_param in zip(self.critic.q1.parameters(), self.critic.q1_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                    for param, target_param in zip(self.critic.q2.parameters(), self.critic.q2_target.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                    # Policy Target Update (Specific to MPO, often slower than critic)
                    for param, target_param in zip(self.policy.parameters(), self.target_policy.parameters()):
                        target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

                # Metrics
                optimization_metrics = {
                    "dual/eta": eta.item(),
                    "dual/alpha_mean_avg": alpha_mean.mean().item(),
                    "dual/alpha_std_avg": alpha_std.mean().item(),
                    "kl/mean_kl_mean": mean_kl_mean.mean().item(),
                    "kl/mean_kl_std": mean_kl_std.mean().item(),
                    "loss/q_loss": q_loss.item(),
                    "loss/policy_loss": policy_loss.item(),
                    "loss/dual_loss": dual_loss.item(),
                    "q_value/q_mean": q_values.mean().item(),
                    "gradients/policy_grad_norm": policy_grad_norm.item(),
                    "gradients/critic_grad_norm": critic_grad_norm.item(),
                }

                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1
            
            if self.anneal_learning_rate:
                self.q_scheduler.step()
                self.policy_scheduler.step()
                self.dual_scheduler.step()
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)

            # --- Evaluating ---
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        eval_processed_action = self.policy.get_deterministic_action(torch.tensor(eval_state, dtype=torch.float32).to(self.device))
                    
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.cpu().numpy())
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

            # --- Saving ---
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

            # --- Logging ---
            if should_log:
                self.start_logging(global_step)
                
                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes
                
                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    for info_name in list(step_info_collection.keys()):
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(step_info_collection[info_name])
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}
                self.end_logging()
            
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time

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
        file_path = self.save_path + "/best.model"
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.critic.q1.state_dict(),
            "q2_state_dict": self.critic.q2.state_dict(),
            "q1_target_state_dict": self.critic.q1_target.state_dict(),
            "q2_target_state_dict": self.critic.q2_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "dual_optimizer_state_dict": self.dual_optimizer.state_dict(),
            # Dual variables
            "log_eta": self.log_eta,
            "log_alpha_mean": self.log_alpha_mean,
            "log_alpha_std": self.log_alpha_std,
            "log_penalty_temperature": self.log_penalty_temperature
        }
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))

    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        
        model = MPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.q1.load_state_dict(checkpoint["q1_state_dict"])
        model.critic.q2.load_state_dict(checkpoint["q2_state_dict"])
        model.critic.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        model.critic.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        
        # Load Duals
        model.log_eta.data = checkpoint["log_eta"].data
        model.log_alpha_mean.data = checkpoint["log_alpha_mean"].data
        model.log_alpha_std.data = checkpoint["log_alpha_std"].data
        model.log_penalty_temperature.data = checkpoint["log_penalty_temperature"].data

        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        model.dual_optimizer.load_state_dict(checkpoint["dual_optimizer_state_dict"])
        
        return model

    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")

    def set_train_mode(self):
        self.policy.train()
        self.critic.train()

    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()

    @staticmethod
    def general_properties():
        return GeneralProperties