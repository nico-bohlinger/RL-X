import math
import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.algorithms.reppo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.reppo.pytorch.policy import get_policy
from rl_x.algorithms.reppo.pytorch.critic import get_critic
from rl_x.algorithms.reppo.pytorch.batch import Batch
from rl_x.algorithms.reppo.pytorch.observation_normalizer import ObservationNormalizer

rlx_logger = logging.getLogger("rl_x")


class REPPO:
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
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.nr_minibatches = config.algorithm.nr_minibatches
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.nr_bins = config.algorithm.nr_bins
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.kl_bound = config.algorithm.kl_bound
        self.auxiliary_loss_coefficient = config.algorithm.auxiliary_loss_coefficient
        self.nr_kl_samples = config.algorithm.nr_kl_samples
        self.normalize_observation = config.algorithm.normalize_observation
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_rollout_updates = self.total_timesteps // self.batch_size
        self.minibatch_size = self.batch_size // self.nr_minibatches
        self.action_dimension = self.train_env.single_action_space.shape[0]
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = (self.action_dimension,)
        self.target_entropy = self.action_dimension * config.algorithm.target_entropy_multiplier

        if self.nr_rollout_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.batch_size % self.nr_minibatches != 0:
            raise ValueError("The rollout batch size must be divisible by the number of minibatches.")
        if self.evaluation_frequency != -1 and self.evaluation_frequency % self.batch_size != 0:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

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

        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        self.observation_normalizer = ObservationNormalizer(self.os_shape, self.normalize_observation, self.device)
        self.old_policy = get_policy(config, self.train_env, self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        for p in self.old_policy.parameters():
            p.requires_grad = False

        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate, fused=fused)

        if self.anneal_learning_rate:
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.nr_rollout_updates)
            self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.nr_rollout_updates)

        self.is_torch_data_interface = self.train_env.general_properties.data_interface_type == DataInterfaceType.TORCH

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        bin_width = (self.v_max - self.v_min) / (self.nr_bins - 1)
        hl_gauss_centers = torch.linspace(self.v_min, self.v_max, self.nr_bins, device=self.device, dtype=torch.float32)
        hl_gauss_support = torch.linspace(self.v_min - bin_width / 2, self.v_max + bin_width / 2, self.nr_bins + 1, device=self.device, dtype=torch.float32)
        hl_gauss_sigma = bin_width * 0.75


        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(obs_mb, action_mb, target_mb, reward_mb, next_features_mb, terminated_mb, truncated_mb):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                _, critic_logits, pred_features, pred_reward = self.critic(obs_mb, action_mb)
                cdf_evals = torch.erf((hl_gauss_support - torch.clamp(target_mb, self.v_min, self.v_max).unsqueeze(-1)) / (math.sqrt(2) * hl_gauss_sigma))
                target_dist = (cdf_evals[..., 1:] - cdf_evals[..., :-1]) / (cdf_evals[..., -1:] - cdf_evals[..., :1])
                critic_update_loss = -(target_dist * torch.log_softmax(critic_logits, dim=-1)).sum(-1)
                auxiliary_loss = torch.cat([(pred_features - next_features_mb).square(), (pred_reward - reward_mb.unsqueeze(-1)).square()], dim=-1).mean(-1)
                critic_update_loss = (1.0 - truncated_mb) * critic_update_loss
                auxiliary_loss = (1.0 - truncated_mb) * (1.0 - terminated_mb) * auxiliary_loss
                critic_loss = critic_update_loss.mean() + self.auxiliary_loss_coefficient * auxiliary_loss.mean()
                value = torch.sum(torch.softmax(critic_logits, dim=-1) * hl_gauss_centers, dim=-1)
                explained_variance = 1 - torch.var(target_mb - value, correction=0) / (torch.var(target_mb, correction=0) + 1e-8)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            return critic_update_loss.mean(), auxiliary_loss.mean(), value.mean(), explained_variance, critic_grad_norm


        @torch.compile(mode=self.compile_mode)
        def policy_loss_fn(obs_mb):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                loc, log_std = self.policy(obs_mb)
                new_action, new_log_prob = self.policy.sample_and_log_prob(loc, log_std)
                _, critic_logits_new, _, _ = self.critic(obs_mb, new_action)
                value = torch.sum(torch.softmax(critic_logits_new, dim=-1) * hl_gauss_centers, dim=-1)

                with torch.no_grad():
                    old_loc, old_log_std = self.old_policy(obs_mb)
                old_loc_expanded = old_loc.unsqueeze(0).expand(self.nr_kl_samples, -1, -1)
                old_log_std_expanded = old_log_std.unsqueeze(0).expand(self.nr_kl_samples, -1, -1)
                old_actions, old_log_probs = self.old_policy.sample_and_log_prob(old_loc_expanded, old_log_std_expanded)
                loc_expanded = loc.unsqueeze(0).expand(self.nr_kl_samples, -1, -1)
                log_std_expanded = log_std.unsqueeze(0).expand(self.nr_kl_samples, -1, -1)
                new_log_probs_at_old = self.policy.log_prob(loc_expanded, log_std_expanded, old_actions)
                kl = (old_log_probs - new_log_probs_at_old).mean(0)

                entropy_coefficient = self.policy.log_entropy_coefficient.exp().squeeze()
                kl_coefficient = self.policy.log_kl_coefficient.exp().squeeze()
                clipped_loss = torch.where(
                    kl < self.kl_bound,
                    new_log_prob * entropy_coefficient.detach() - value,
                    kl * kl_coefficient.detach(),
                )
                entropy = -new_log_prob
                target_entropy_value = self.target_entropy + entropy
                entropy_coefficient_loss = entropy_coefficient * target_entropy_value.detach()
                kl_coefficient_loss = -kl_coefficient * (kl.detach() - self.kl_bound)
                policy_loss = clipped_loss.mean() + entropy_coefficient_loss.mean() + kl_coefficient_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            return (
                clipped_loss.mean(),
                entropy_coefficient_loss.mean(),
                kl_coefficient_loss.mean(),
                entropy.mean(),
                kl.mean(),
                entropy_coefficient.detach(),
                kl_coefficient.detach(),
                value.mean(),
                policy_grad_norm,
            )


        @torch.compile(mode=self.compile_mode)
        def rollout_act(state):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                loc, log_std = self.policy(state)
                action, _ = self.policy.sample_and_log_prob(loc, log_std)
            return action, self.policy.get_processed_action(action)


        @torch.compile(mode=self.compile_mode)
        def rollout_evaluate_next(actual_next_state, reward_t):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                next_loc, next_log_std = self.policy(actual_next_state)
                next_action, next_log_prob = self.policy.sample_and_log_prob(next_loc, next_log_std)
                next_features, next_logits, _, _ = self.critic(actual_next_state, next_action)
                next_value = torch.sum(torch.softmax(next_logits, dim=-1) * hl_gauss_centers, dim=-1)
                entropy_coefficient = self.policy.log_entropy_coefficient.exp().squeeze()
                soft_reward = reward_t - self.gamma * next_log_prob * entropy_coefficient
            return next_features, next_value, soft_reward


        @torch.jit.script
        def compute_td_lambda_targets(soft_rewards, next_values, terminations, truncations, gamma: float, gae_lambda: float):
            target_values = torch.zeros_like(next_values)
            lambda_return = next_values[-1]
            for t in range(next_values.shape[0] - 1, -1, -1):
                value_t = next_values[t]
                done_t = terminations[t]
                truncated_t = truncations[t]
                lambda_sum = gae_lambda * lambda_return + (1 - gae_lambda) * value_t
                delta = gamma * (truncated_t * value_t + (1.0 - truncated_t) * (1.0 - done_t) * lambda_sum)
                lambda_return = soft_rewards[t] + delta
                target_values[t] = lambda_return
            return target_values


        self.set_train_mode()

        batch = Batch(
            states=torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device),
            actions=torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32, device=self.device),
            rewards=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            soft_rewards=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            next_features=torch.zeros((self.nr_steps, self.nr_envs, self.config.algorithm.critic_hidden_dim), dtype=torch.float32, device=self.device),
            next_values=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            terminations=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            truncations=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        if not self.is_torch_data_interface:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
            if logging_time_prev:
                time_metrics["time/logging_time_prev"] = logging_time_prev


            # Acting
            with torch.no_grad():
                dones_this_rollout = 0
                step_info_collection = {}
                for step in range(self.nr_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    self.observation_normalizer.update(state)
                    normalized_state = self.observation_normalizer.normalize(state)
                    action, processed_action = rollout_act(normalized_state)
                    action_to_env = processed_action if self.is_torch_data_interface else processed_action.cpu().numpy()

                    next_state, reward, terminated, truncated, info = self.train_env.step(action_to_env)
                    done = terminated | truncated
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
                        terminated_t = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                        truncated_t = torch.tensor(truncated, dtype=torch.float32, device=self.device)
                    else:
                        actual_next_state = next_state
                        dones_this_step = int(done.sum().item())
                        dones_this_rollout += dones_this_step
                        reward_t = reward.float()
                        terminated_t = terminated.float()
                        truncated_t = truncated.float()
                        if dones_this_step > 0:
                            environment_state = getattr(self.train_env, "env_state", None)
                            if environment_state is not None and hasattr(environment_state, "actual_next_observation"):
                                actual_next_observation = environment_state.actual_next_observation
                                actual_next_state = actual_next_observation if isinstance(actual_next_observation, torch.Tensor) else torch.from_dlpack(actual_next_observation)
                            elif "final_observation" in info and isinstance(info["final_observation"], torch.Tensor):
                                actual_next_state = next_state.clone()
                                final_observation_mask = info.get("_final_observation", done).bool()
                                actual_next_state[final_observation_mask] = info["final_observation"][final_observation_mask]
                            else:
                                terminated_t = done.float()
                                truncated_t = torch.zeros_like(terminated_t)
                    logging_info = self.train_env.get_logging_info_dict(info)
                    if self.is_torch_data_interface and dones_this_step > 0:
                        episode_returns = logging_info.get("episode_return", logging_info.get("rollout/episode_return"))
                        if episode_returns is not None:
                            if isinstance(episode_returns, torch.Tensor):
                                episode_returns = episode_returns.detach().cpu().numpy()
                            done_indices = done.detach().cpu().numpy().astype(bool)
                            episode_returns = np.asarray(episode_returns).reshape(-1)
                            if episode_returns.size == self.nr_envs:
                                episode_returns = episode_returns[done_indices]
                            if episode_returns.size == dones_this_step:
                                saving_return_buffer.extend(episode_returns.tolist())
                    for k, info_value in logging_info.items():
                        if isinstance(info_value, torch.Tensor):
                            info_value = info_value.detach().cpu().numpy()
                        step_info_collection.setdefault(k, []).extend(np.asarray(info_value).reshape(-1).tolist())

                    torch.compiler.cudagraph_mark_step_begin()
                    normalized_actual_next_state = self.observation_normalizer.normalize(actual_next_state)
                    next_features, next_value, soft_reward = rollout_evaluate_next(normalized_actual_next_state, reward_t)

                    batch.states[step] = normalized_state
                    batch.actions[step] = action
                    batch.rewards[step] = reward_t
                    batch.soft_rewards[step] = soft_reward
                    batch.next_features[step] = next_features
                    batch.next_values[step] = next_value
                    batch.terminations[step] = terminated_t
                    batch.truncations[step] = truncated_t
                    state = next_state
                    global_step += self.nr_envs
                nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating TD-lambda targets
            with torch.no_grad():
                target_values = compute_td_lambda_targets(batch.soft_rewards, batch.next_values, batch.terminations, batch.truncations, self.gamma, self.gae_lambda)

            calc_target_end_time = time.time()
            time_metrics["time/calc_target_time"] = calc_target_end_time - acting_end_time


            # Snapshot old policy
            self.old_policy.load_state_dict(self.policy.state_dict())


            # Optimizing
            batch_states_flat = batch.states.reshape((-1,) + self.os_shape)
            batch_actions_flat = batch.actions.reshape((-1,) + self.as_shape)
            batch_rewards_flat = batch.rewards.reshape(-1)
            batch_targets_flat = target_values.reshape(-1)
            batch_next_features_flat = batch.next_features.reshape((-1, batch.next_features.shape[-1]))
            batch_terminations_flat = batch.terminations.reshape(-1)
            batch_truncations_flat = batch.truncations.reshape(-1)

            optimization_metrics_list = []
            batch_indices = np.arange(self.batch_size)
            for epoch in range(self.nr_epochs):
                self.rng.shuffle(batch_indices)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb = torch.tensor(batch_indices[start:end], dtype=torch.long, device=self.device)
                    obs_mb = batch_states_flat[mb]
                    action_mb = batch_actions_flat[mb]
                    reward_mb = batch_rewards_flat[mb]
                    target_mb = batch_targets_flat[mb]
                    next_features_mb = batch_next_features_flat[mb]
                    terminated_mb = batch_terminations_flat[mb]
                    truncated_mb = batch_truncations_flat[mb]

                    critic_update_loss_mean, auxiliary_loss_mean, q_mean, explained_variance, critic_grad_norm = critic_loss_fn(
                        obs_mb, action_mb, target_mb, reward_mb, next_features_mb, terminated_mb, truncated_mb,
                    )
                    for parameter in self.critic.parameters():
                        parameter.requires_grad = False
                    policy_loss_mean, entropy_coefficient_loss_mean, kl_coefficient_loss_mean, entropy_mean, kl_mean, entropy_coefficient, kl_coefficient, policy_q_mean, policy_grad_norm = policy_loss_fn(obs_mb)
                    for parameter in self.critic.parameters():
                        parameter.requires_grad = True

                    optimization_metrics_list.append({
                        "loss/critic_loss": critic_update_loss_mean.item(),
                        "loss/auxiliary_loss": auxiliary_loss_mean.item(),
                        "loss/policy_loss": policy_loss_mean.item(),
                        "loss/entropy_coefficient_loss": entropy_coefficient_loss_mean.item(),
                        "loss/kl_coefficient_loss": kl_coefficient_loss_mean.item(),
                        "entropy/entropy": entropy_mean.item(),
                        "entropy/entropy_coefficient": entropy_coefficient.item(),
                        "kl/kl_divergence": kl_mean.item(),
                        "kl/kl_coefficient": kl_coefficient.item(),
                        "q/q_mean": q_mean.item(),
                        "q/policy_q_mean": policy_q_mean.item(),
                        "q/explained_variance": explained_variance.item(),
                        "gradients/policy_grad_norm": policy_grad_norm.item(),
                        "gradients/critic_grad_norm": critic_grad_norm.item(),
                    })

            if self.anneal_learning_rate:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            optimization_metrics = {k: np.mean([m[k] for m in optimization_metrics_list]) for k in optimization_metrics_list[0].keys()}
            optimization_metrics["lr/learning_rate"] = self.learning_rate if not self.anneal_learning_rate else self.policy_scheduler.get_last_lr()[0]
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_target_end_time


            # Evaluating
            evaluation_metrics = {}
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                with torch.inference_mode():
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    if not self.is_torch_data_interface:
                        eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                    eval_nr_episodes = 0
                    evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                    eval_episode_returns = np.zeros(self.nr_envs, dtype=np.float64)
                    eval_episode_lengths = np.zeros(self.nr_envs, dtype=np.int64)
                    while True:
                        torch.compiler.cudagraph_mark_step_begin()
                        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                            normalized_eval_state = self.observation_normalizer.normalize(eval_state)
                            eval_loc, _ = self.policy(normalized_eval_state)
                            eval_action = self.policy.get_processed_action(self.policy.deterministic_action(eval_loc))
                        eval_action_to_env = eval_action if self.is_torch_data_interface else eval_action.cpu().numpy()
                        eval_state, eval_reward, eval_terminated, eval_truncated, _ = self.eval_env.step(eval_action_to_env)
                        if not self.is_torch_data_interface:
                            eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                            eval_done = eval_terminated | eval_truncated
                        else:
                            eval_reward = eval_reward.detach().cpu().numpy()
                            eval_done = (eval_terminated | eval_truncated).detach().cpu().numpy()
                        eval_episode_returns += np.asarray(eval_reward)
                        eval_episode_lengths += 1
                        for i, single_done in enumerate(eval_done):
                            if single_done:
                                eval_nr_episodes += 1
                                evaluation_metrics["eval/episode_return"].append(eval_episode_returns[i])
                                evaluation_metrics["eval/episode_length"].append(eval_episode_lengths[i])
                                eval_episode_returns[i] = 0.0
                                eval_episode_lengths[i] = 0
                                if eval_nr_episodes == self.evaluation_episodes:
                                    break
                        if eval_nr_episodes == self.evaluation_episodes:
                            break
                    self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            if self.save_model and dones_this_rollout > 0 and saving_return_buffer:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            rollout_info_metrics = {}
            env_info_metrics = {}
            if step_info_collection:
                info_names = list(step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(step_info_collection[info_name])
                    if mean_value == mean_value:
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value

            evaluation_metrics = {k: np.mean(v) for k, v in evaluation_metrics.items()}

            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

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
        checkpoint = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "observation_normalizer_state_dict": self.observation_normalizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        if self.anneal_learning_rate:
            checkpoint["policy_scheduler_state_dict"] = self.policy_scheduler.state_dict()
            checkpoint["critic_scheduler_state_dict"] = self.critic_scheduler.state_dict()
        torch.save(checkpoint, file_path)
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
        model = REPPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.observation_normalizer.load_state_dict(checkpoint["observation_normalizer_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        if model.anneal_learning_rate:
            model.policy_scheduler.load_state_dict(checkpoint["policy_scheduler_state_dict"])
            model.critic_scheduler.load_state_dict(checkpoint["critic_scheduler_state_dict"])
        return model


    def test(self, episodes):
        with torch.inference_mode():
            self.set_eval_mode()
            for i in range(episodes):
                done = False
                episode_return = 0
                state, _ = self.eval_env.reset()
                while not done:
                    torch.compiler.cudagraph_mark_step_begin()
                    if not self.is_torch_data_interface:
                        state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        normalized_state = self.observation_normalizer.normalize(state)
                        loc, _ = self.policy(normalized_state)
                        action = self.policy.get_processed_action(self.policy.deterministic_action(loc))
                    action_to_env = action if self.is_torch_data_interface else action.cpu().numpy()
                    state, reward, terminated, truncated, info = self.eval_env.step(action_to_env)
                    done = terminated | truncated
                    episode_return += reward
                rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()
        self.observation_normalizer.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()
        self.observation_normalizer.eval()


    def general_properties():
        return GeneralProperties
