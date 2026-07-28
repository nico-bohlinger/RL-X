import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.fastmpo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.fastmpo.pytorch.policy import get_policy
from rl_x.algorithms.fastmpo.pytorch.critic import get_critic
from rl_x.algorithms.fastmpo.pytorch.dual_variables import DualVariables
from rl_x.algorithms.fastmpo.pytorch.observation_normalizer import ObservationNormalizer
from rl_x.algorithms.fastmpo.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class FastMPO:
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
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.dual_critic = config.algorithm.dual_critic
        self.action_clipping = config.algorithm.action_clipping
        self.policy_learning_rate = config.algorithm.policy_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_policy_learning_rate = config.algorithm.anneal_policy_learning_rate
        self.anneal_critic_learning_rate = config.algorithm.anneal_critic_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.policy_weight_decay = config.algorithm.policy_weight_decay
        self.critic_weight_decay = config.algorithm.critic_weight_decay
        self.dual_weight_decay = config.algorithm.dual_weight_decay
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.collect_data_with_online_policy = config.algorithm.collect_data_with_online_policy
        self.action_sampling_number = config.algorithm.action_sampling_number
        self.epsilon_non_parametric = config.algorithm.epsilon_non_parametric
        self.epsilon_parametric_mu = config.algorithm.epsilon_parametric_mu
        self.epsilon_parametric_sigma = config.algorithm.epsilon_parametric_sigma
        self.epsilon_penalty = config.algorithm.epsilon_penalty
        self.float_epsilon = config.algorithm.float_epsilon
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.critic_tau = config.algorithm.critic_tau
        self.policy_tau = config.algorithm.policy_tau
        self.gamma = config.algorithm.gamma
        self.nr_atoms = config.algorithm.nr_atoms
        self.n_steps = config.algorithm.n_steps
        self.clipped_double_q_learning = config.algorithm.clipped_double_q_learning
        self.nr_critic_updates_per_policy_update = config.algorithm.nr_critic_updates_per_policy_update
        self.nr_policy_updates_per_step = config.algorithm.nr_policy_updates_per_step
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.horizon = self.train_env.horizon

        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")
        if self.evaluation_and_save_frequency != -1 and self.evaluation_and_save_frequency % self.nr_envs != 0:
            raise ValueError("The evaluation and save frequency must be a multiple of the number of environments.")
        if self.learning_starts < self.n_steps:
            raise ValueError("The replay buffer must contain at least n_steps transitions before learning starts.")
        if self.nr_parallel_seeds != 1:
            raise ValueError("Parallel seeds are only supported by the fully JIT-compiled implementation.")
        if self.clipped_double_q_learning and not self.dual_critic:
            raise ValueError("Clipped double Q-learning requires two critics.")

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

        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.policy = get_policy(config, self.train_env, self.device)
        self.policy_target = get_policy(config, self.train_env, self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        for parameter in self.policy_target.parameters():
            parameter.requires_grad_(False)
        self.critic = get_critic(config, self.train_env, self.device)
        nr_actions = np.prod(self.train_env.single_action_space.shape).item()
        self.dual_variables = DualVariables(nr_actions, config.algorithm.init_log_eta, config.algorithm.init_log_alpha_mean, config.algorithm.init_log_alpha_stddev, config.algorithm.init_log_penalty_temperature).to(self.device)
        self.observation_normalizer = ObservationNormalizer(self.train_env.single_observation_space.shape[0], config.algorithm.enable_observation_normalization, config.algorithm.normalizer_epsilon).to(self.device)
        self.q_support = torch.linspace(self.v_min, self.v_max, self.nr_atoms, device=self.device)

        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.policy_learning_rate, weight_decay=config.algorithm.policy_weight_decay, betas=(self.adam_beta1, self.adam_beta2), fused=fused)
        self.critic_optimizer = optim.AdamW(self.critic.q_networks.parameters(), lr=self.critic_learning_rate, weight_decay=config.algorithm.critic_weight_decay, betas=(self.adam_beta1, self.adam_beta2), fused=fused)
        self.dual_optimizer = optim.AdamW(self.dual_variables.parameters(), lr=self.dual_learning_rate, weight_decay=config.algorithm.dual_weight_decay, betas=(self.adam_beta1, self.adam_beta2), fused=fused)
        total_iterations = max(1, self.total_timesteps // self.nr_envs - self.learning_starts)
        if self.anneal_policy_learning_rate:
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iterations)
        if self.anneal_critic_learning_rate:
            self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iterations)
        if self.anneal_dual_learning_rate:
            self.dual_scheduler = optim.lr_scheduler.LinearLR(self.dual_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iterations)

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)


    def train(self):
        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, dones, truncations, effective_n_steps):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_action_mean, next_action_std = self.policy_target(next_states)
                    sampled_next_actions_raw = next_action_mean.unsqueeze(0) + next_action_std.unsqueeze(0) * torch.randn((self.action_sampling_number,) + next_action_mean.shape, device=self.device)
                    sampled_next_actions = torch.clamp(sampled_next_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_next_actions_raw
                    expanded_next_states = next_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)
                    target_logits = self.critic.forward_target(expanded_next_states.reshape((-1, expanded_next_states.shape[-1])), sampled_next_actions.reshape((-1, sampled_next_actions.shape[-1])))
                    target_pmf = F.softmax(target_logits.reshape((target_logits.shape[0], self.action_sampling_number, states.shape[0], self.nr_atoms)), dim=-1).mean(dim=1)
                    discount = self.gamma ** effective_n_steps * (1.0 - dones * (1.0 - truncations))
                    target_z = torch.clamp(rewards[:, None] + discount[:, None] * self.q_support[None], self.v_min, self.v_max)
                    b = (target_z - self.v_min) / ((self.v_max - self.v_min) / (self.nr_atoms - 1))
                    lower = torch.floor(b).long()
                    upper = torch.ceil(b).long()
                    is_integer = upper == lower
                    lower = torch.where(is_integer & (lower > 0), lower - 1, lower)
                    upper = torch.where(is_integer & (upper == 0), upper + 1, upper)
                    lower_weight = upper.float() - b
                    upper_weight = b - lower.float()
                    projected = torch.zeros_like(target_pmf)
                    projected.scatter_add_(2, lower.unsqueeze(0).expand(target_pmf.shape[0], -1, -1), target_pmf * lower_weight.unsqueeze(0))
                    projected.scatter_add_(2, upper.unsqueeze(0).expand(target_pmf.shape[0], -1, -1), target_pmf * upper_weight.unsqueeze(0))
                    target_values = torch.sum(projected * self.q_support[None, None], dim=-1)
                    if self.dual_critic and self.clipped_double_q_learning:
                        chosen = torch.where((target_values[0] <= target_values[1])[:, None], projected[0], projected[1])
                        target_distribution = chosen.unsqueeze(0).expand(projected.shape[0], -1, -1)
                    else:
                        target_distribution = projected
                actions = torch.clamp(actions, -1.0, 1.0) if self.action_clipping else actions
                current_logits = self.critic(states, actions)
                q_loss = -torch.sum(target_distribution * F.log_softmax(current_logits, dim=-1), dim=(0, 2)).mean()

            self.critic_optimizer.zero_grad()
            q_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.q_networks.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            return q_loss, target_values.mean(), target_values.max(), target_values.min(), critic_grad_norm


        @torch.compile(mode=self.compile_mode)
        def policy_and_dual_loss_fn(states, next_states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                stacked_states = torch.stack([states, next_states], dim=1)
                with torch.no_grad():
                    target_action_mean, target_action_std = self.policy_target(stacked_states)
                    sampled_actions_raw = target_action_mean.unsqueeze(0) + target_action_std.unsqueeze(0) * torch.randn((self.action_sampling_number,) + target_action_mean.shape, device=self.device)
                    sampled_actions = torch.clamp(sampled_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_actions_raw
                    expanded_states = stacked_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1, -1)
                    logits = self.critic.forward_target(expanded_states.reshape((-1, expanded_states.shape[-1])), sampled_actions.reshape((-1, sampled_actions.shape[-1])))
                    q_values = torch.sum(F.softmax(logits, dim=-1) * self.q_support[None, None], dim=-1).reshape((logits.shape[0], self.action_sampling_number, states.shape[0], 2))
                    if q_values.shape[0] == 1:
                        q_values = q_values[0]
                    elif self.clipped_double_q_learning:
                        q_values = torch.min(q_values, dim=0).values
                    else:
                        q_values = torch.mean(q_values, dim=0)

                log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature = self.dual_variables()
                eta = F.softplus(log_eta)[0] + self.float_epsilon
                improvement_distribution = F.softmax(q_values / eta.detach(), dim=0)
                loss_eta = eta * (self.epsilon_non_parametric + torch.logsumexp(q_values / eta, dim=0).mean() - np.log(self.action_sampling_number))
                if self.action_clipping:
                    penalty_temperature = F.softplus(log_penalty_temperature)[0] + self.float_epsilon
                    out_of_bounds_cost = -torch.linalg.vector_norm(sampled_actions_raw - torch.clamp(sampled_actions_raw, -1.0, 1.0), dim=-1)
                    improvement_distribution = improvement_distribution + F.softmax(out_of_bounds_cost / penalty_temperature.detach(), dim=0)
                    loss_eta = loss_eta + penalty_temperature * (self.epsilon_penalty + torch.logsumexp(out_of_bounds_cost / penalty_temperature, dim=0).mean() - np.log(self.action_sampling_number))
                else:
                    penalty_temperature = torch.zeros((), device=self.device)

                online_action_mean, online_action_std = self.policy(stacked_states)
                alpha_mean = F.softplus(log_alpha_mean) + self.float_epsilon
                alpha_std = F.softplus(log_alpha_stddev) + self.float_epsilon
                logprob_mean = (-0.5 * (((sampled_actions_raw - online_action_mean) / target_action_std) ** 2 + np.log(2.0 * np.pi)) - torch.log(target_action_std)).sum(dim=-1)
                loss_pg_mean = -(logprob_mean * improvement_distribution).sum(dim=0).mean()
                target_action_std_clipped = torch.clamp(target_action_std, min=self.float_epsilon)
                mean_kl_mean = (((target_action_mean - online_action_mean) ** 2) / (2.0 * target_action_std_clipped ** 2)).mean(dim=(0, 1))
                loss_kl_mean = torch.sum(alpha_mean.detach() * mean_kl_mean)
                loss_alpha_mean = torch.sum(alpha_mean * (self.epsilon_parametric_mu - mean_kl_mean.detach()))

                logprob_std = (-0.5 * (((sampled_actions_raw - target_action_mean) / online_action_std) ** 2 + np.log(2.0 * np.pi)) - torch.log(online_action_std)).sum(dim=-1)
                loss_pg_std = -(logprob_std * improvement_distribution).sum(dim=0).mean()
                online_action_std_clipped = torch.clamp(online_action_std, min=self.float_epsilon)
                mean_kl_std = (torch.log(online_action_std_clipped / target_action_std_clipped) + target_action_std_clipped ** 2 / (2.0 * online_action_std_clipped ** 2) - 0.5).mean(dim=(0, 1))
                loss_kl_std = torch.sum(alpha_std.detach() * mean_kl_std)
                loss_alpha_std = torch.sum(alpha_std * (self.epsilon_parametric_sigma - mean_kl_std.detach()))
                actor_loss = loss_pg_mean + loss_pg_std + loss_kl_mean + loss_kl_std
                dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta

            self.policy_optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            (actor_loss + dual_loss).backward()
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            dual_grad_norm = torch.nn.utils.clip_grad_norm_(self.dual_variables.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            self.dual_optimizer.step()
            return actor_loss, loss_pg_mean, loss_pg_std, loss_kl_mean, loss_kl_std, dual_loss, loss_alpha_mean, loss_alpha_std, loss_eta, eta, penalty_temperature, alpha_mean.mean(), alpha_std.mean(), mean_kl_mean.mean(), mean_kl_std.mean(), q_values.mean(), online_action_std.min(dim=-1).values.mean(), online_action_std.max(dim=-1).values.mean(), policy_grad_norm, dual_grad_norm


        self.set_train_mode()
        replay_buffer = ReplayBuffer(self.buffer_size_per_env, self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma, self.device)
        state, _ = self.train_env.reset()
        global_step = 0
        nr_critic_updates = 0
        nr_policy_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            torch.compiler.cudagraph_mark_step_begin()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)

            # Acting
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                normalized_state = self.observation_normalizer.normalize(state)
                acting_policy = self.policy if self.collect_data_with_online_policy else self.policy_target
                action, processed_action = acting_policy.get_action(normalized_state)
            next_state, reward, terminated, truncated, info = self.train_env.step(processed_action)
            done = terminated | truncated
            actual_next_state = next_state.clone()
            dones_this_rollout = 0
            for index, single_done in enumerate(done):
                if single_done:
                    actual_next_state[index] = self.train_env.get_final_observation_at_index(info, index)
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)
            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)

            should_optimize = global_step > self.learning_starts * self.nr_envs
            should_evaluate = self.evaluation_active and self.evaluation_and_save_frequency != -1 and global_step % self.evaluation_and_save_frequency == 0
            should_save = should_optimize and self.save_model and self.evaluation_and_save_frequency != -1 and global_step % self.evaluation_and_save_frequency == 0
            should_log = global_step % self.logging_frequency == 0

            # Optimizing
            if should_optimize:
                total_critic_updates = self.nr_policy_updates_per_step * self.nr_critic_updates_per_policy_update
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations, batch_effective_n_steps = replay_buffer.sample(total_critic_updates * self.batch_size)
                self.observation_normalizer.normalize(torch.cat([batch_states, batch_next_states]), update=True)
                normalized_states = self.observation_normalizer.normalize(batch_states).reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                normalized_next_states = self.observation_normalizer.normalize(batch_next_states).reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_actions = batch_actions.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_rewards = batch_rewards.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_dones = batch_dones.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_truncations = batch_truncations.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_effective_n_steps = batch_effective_n_steps.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)

                for policy_update in range(self.nr_policy_updates_per_step):
                    for critic_update in range(self.nr_critic_updates_per_policy_update):
                        metrics = critic_loss_fn(normalized_states[policy_update, critic_update], normalized_next_states[policy_update, critic_update], batch_actions[policy_update, critic_update], batch_rewards[policy_update, critic_update], batch_dones[policy_update, critic_update], batch_truncations[policy_update, critic_update], batch_effective_n_steps[policy_update, critic_update])
                        metric_names = ["loss/q_loss", "q/q_mean", "q/q_max", "q/q_min", "gradients/critic_grad_norm"]
                        for key, value in zip(metric_names, metrics):
                            optimization_metrics_collection.setdefault(key, []).append(value.detach().clone())
                        with torch.no_grad():
                            for target_parameter, parameter in zip(self.critic.target_q_networks.parameters(), self.critic.q_networks.parameters()):
                                target_parameter.lerp_(parameter, self.critic_tau)
                        nr_critic_updates += 1

                    metrics = policy_and_dual_loss_fn(normalized_states[policy_update, -1], normalized_next_states[policy_update, -1])
                    metric_names = ["loss/actor_loss", "loss/loss_pg_mean", "loss/loss_pg_std", "loss/loss_kl_mean", "loss/loss_kl_std", "loss/dual_loss", "loss/loss_alpha_mean", "loss/loss_alpha_std", "loss/loss_eta", "dual/eta", "dual/penalty_temperature", "dual/alpha_mean", "dual/alpha_std", "kl/mean_kl_mean", "kl/mean_kl_std", "q/improvement_q_mean", "policy/std_min_mean", "policy/std_max_mean", "gradients/policy_grad_norm", "gradients/dual_variables_grad_norm"]
                    for key, value in zip(metric_names, metrics):
                        optimization_metrics_collection.setdefault(key, []).append(value.detach().clone())
                    with torch.no_grad():
                        self.dual_variables.log_eta.clamp_(min=self.min_log_temperature)
                        self.dual_variables.log_alpha_mean.clamp_(min=self.min_log_alpha)
                        self.dual_variables.log_alpha_stddev.clamp_(min=self.min_log_alpha)
                        for target_parameter, parameter in zip(self.policy_target.parameters(), self.policy.parameters()):
                            target_parameter.lerp_(parameter, self.policy_tau)
                    nr_policy_updates += 1

                if self.anneal_policy_learning_rate:
                    self.policy_scheduler.step()
                if self.anneal_critic_learning_rate:
                    self.critic_scheduler.step()
                if self.anneal_dual_learning_rate:
                    self.dual_scheduler.step()
                optimization_metrics_collection.setdefault("lr/policy_learning_rate", []).append(self.policy_optimizer.param_groups[0]["lr"])
                optimization_metrics_collection.setdefault("lr/critic_learning_rate", []).append(self.critic_optimizer.param_groups[0]["lr"])
                optimization_metrics_collection.setdefault("lr/dual_variables_learning_rate", []).append(self.dual_optimizer.param_groups[0]["lr"])
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)

            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                for _ in range(self.horizon):
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        eval_action = self.policy.get_deterministic_action(self.observation_normalizer.normalize(eval_state))
                    eval_state, _, _, _, eval_info = self.eval_env.step(eval_action)
                    eval_logging_info = self.eval_env.get_logging_info_dict(eval_info)
                    if "episode_return" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_return", []).extend(eval_logging_info["episode_return"])
                    if "episode_length" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_length", []).extend(eval_logging_info["episode_length"])
                self.set_train_mode()
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)

            # Saving
            if should_save:
                self.save()
            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            # Logging
            if should_log:
                self.start_logging(global_step)
                rollout_info_metrics = {}
                env_info_metrics = {}
                for info_name, values in step_info_collection.items():
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(values)
                    if mean_value == mean_value:
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: torch.stack(value).float().mean().item() if isinstance(value[0], torch.Tensor) else np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                steps_metrics = {"steps/nr_env_steps": global_step, "steps/nr_critic_updates": nr_critic_updates, "steps/nr_policy_updates": nr_policy_updates, "steps/nr_episodes": nr_episodes}
                for key, value in {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}.items():
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


    def end_logging(self):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        file_path = f"{self.save_path}/latest.model"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "policy_target_state_dict": self.policy_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "dual_variables_state_dict": self.dual_variables.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "dual_optimizer_state_dict": self.dual_optimizer.state_dict(),
            "observation_normalizer_state_dict": self.observation_normalizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        for key, value in checkpoint["config_algorithm"].items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FastMPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.policy_target.load_state_dict(checkpoint["policy_target_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.dual_variables.load_state_dict(checkpoint["dual_variables_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.dual_optimizer.load_state_dict(checkpoint["dual_optimizer_state_dict"])
        model.observation_normalizer.load_state_dict(checkpoint["observation_normalizer_state_dict"])
        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")
        self.set_eval_mode()
        state, _ = self.eval_env.reset()
        while True:
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                action = self.policy.get_deterministic_action(self.observation_normalizer.normalize(state))
            state, _, _, _, _ = self.eval_env.step(action)


    def set_train_mode(self):
        self.policy.train()
        self.policy_target.train()
        self.critic.train()
        self.dual_variables.train()
        self.observation_normalizer.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.policy_target.eval()
        self.critic.eval()
        self.dual_variables.eval()
        self.observation_normalizer.eval()


    def general_properties():
        return GeneralProperties
