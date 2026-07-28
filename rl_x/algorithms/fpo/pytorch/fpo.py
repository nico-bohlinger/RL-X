import os
import logging
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.fpo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.fpo.pytorch.policy import get_policy
from rl_x.algorithms.fpo.pytorch.critic import get_critic
from rl_x.algorithms.fpo.pytorch import observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class FPO:
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
        self.weight_decay = config.algorithm.weight_decay
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clipping_epsilon = config.algorithm.clipping_epsilon
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.reward_scaling = config.algorithm.reward_scaling
        self.normalize_observation = config.algorithm.normalize_observation
        self.observation_normalizer_epsilon = config.algorithm.observation_normalizer_epsilon
        self.observation_normalizer_max_count = config.algorithm.observation_normalizer_max_count
        self.flow_steps = config.algorithm.flow_steps
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.actor_scale = config.algorithm.actor_scale
        self.action_clip = config.algorithm.action_clip
        self.nr_flow_samples_per_action = config.algorithm.nr_flow_samples_per_action
        self.timestep_inverse_cdf_beta = config.algorithm.timestep_inverse_cdf_beta
        self.action_perturb_std = config.algorithm.action_perturb_std
        self.cfm_loss_clamp = config.algorithm.cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages_max = config.algorithm.cfm_loss_clamp_negative_advantages_max
        self.cfm_difference_clamp_max = config.algorithm.cfm_difference_clamp_max
        self.trust_region_mode = config.algorithm.trust_region_mode
        self.advantage_clamp = config.algorithm.advantage_clamp
        self.ema_decay = config.algorithm.ema_decay
        self.ema_warmup_steps = config.algorithm.ema_warmup_steps
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.batch_size = self.nr_envs * self.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("The rollout batch size must be divisible by the minibatch size.")
        if self.flow_steps < 1:
            raise ValueError("Flow steps must be positive.")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be positive and divisible by two.")
        if self.nr_flow_samples_per_action < 1:
            raise ValueError("The number of flow samples per action must be positive.")
        if self.timestep_inverse_cdf_beta <= 0.0:
            raise ValueError("The timestep inverse-CDF beta must be positive.")
        if self.observation_normalizer_epsilon < 0.0 or self.observation_normalizer_max_count <= 0:
            raise ValueError("Observation normalizer epsilon and maximum count are invalid.")
        if self.actor_scale <= 0.0:
            raise ValueError("Actor scale must be positive.")
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0 or self.ema_warmup_steps < 0:
            raise ValueError("EMA decay and warmup are invalid.")
        if self.trust_region_mode not in ["ppo", "spo", "aspo"]:
            raise ValueError("Trust-region mode must be ppo, spo or aspo.")

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
        self.ema_policy = deepcopy(self.policy).to(self.device)
        self.ema_policy.requires_grad_(False)
        self.policy.forward = torch.compile(self.policy.forward, mode=self.compile_mode)
        self.critic.forward = torch.compile(self.critic.forward, mode=self.compile_mode)
        self.ema_policy.forward = torch.compile(self.ema_policy.forward, mode=self.compile_mode)
        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.weight_decay, fused=fused)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.learning_rate, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.weight_decay, fused=fused)
        if self.anneal_learning_rate:
            steps_per_update = self.nr_minibatches * self.nr_epochs
            self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lambda count: max(0.0, 1.0 - (count // steps_per_update) / self.nr_updates))
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda count: max(0.0, 1.0 - (count // steps_per_update) / self.nr_updates))
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)
        self.completed_updates = 0
        self.schedule_current = torch.linspace(1.0, 0.0, self.flow_steps + 1, device=self.device)[:-1]
        self.schedule_next = torch.linspace(1.0, 0.0, self.flow_steps + 1, device=self.device)[1:]

        if self.save_model:
            os.makedirs(self.save_path)


    def normalize(self, observation):
        if self.normalize_observation:
            return observation_normalizer.normalize_observation(self.observation_normalizer_state, observation, self.observation_normalizer_epsilon)
        return observation


    def compute_cfm_loss(self, policy, normalized_observation, action, epsilon, timestep):
        sample_shape = action.shape[:-1] + (self.nr_flow_samples_per_action,)
        observation = torch.broadcast_to(normalized_observation[..., None, :], sample_shape + (normalized_observation.shape[-1],))
        scaled_action = action / self.actor_scale
        noisy_action = timestep * epsilon + (1.0 - timestep) * scaled_action[..., None, :]
        network_prediction = policy(observation, noisy_action, timestep)
        target = epsilon - scaled_action[..., None, :]
        return torch.sum((network_prediction - target) ** 2, dim=-1) / np.sqrt(self.action_dimension)


    def sample_action(self, policy, observation, deterministic=False):
        normalized_observation = torch.tensor(self.normalize(observation), dtype=torch.float32, device=self.device)
        initial_action = torch.randn(observation.shape[:-1] + self.as_shape, device=self.device)
        action = initial_action
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            for current_timestep, next_timestep in zip(self.schedule_current, self.schedule_next):
                timestep = torch.full(observation.shape[:-1] + (1,), current_timestep.item(), device=self.device)
                action = action + (next_timestep - current_timestep) * policy(normalized_observation, action, timestep)
        action = action * self.actor_scale
        if not deterministic:
            action = action + self.action_perturb_std * torch.randn_like(action)
        loss_shape = observation.shape[:-1] + (self.nr_flow_samples_per_action,)
        epsilon = torch.randn(loss_shape + self.as_shape, device=self.device)
        uniform_timestep = torch.rand(loss_shape + (1,), device=self.device)
        timestep = 0.005 + 0.99 * (1.0 - (1.0 - uniform_timestep) ** (1.0 / self.timestep_inverse_cdf_beta))
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            initial_statistic = self.compute_cfm_loss(policy, normalized_observation, action, epsilon, timestep)
        return action, torch.clamp(action, -self.action_clip, self.action_clip), (epsilon, timestep, initial_statistic)


    def train(self):
        self.set_train_mode()
        state, unused_info = self.train_env.reset()
        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device)
            next_states = torch.zeros_like(states)
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32, device=self.device)
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device)
            values = torch.zeros_like(rewards)
            terminations = torch.zeros_like(rewards)
            truncations = torch.zeros_like(rewards)
            epsilon = torch.zeros((self.nr_steps, self.nr_envs, self.nr_flow_samples_per_action) + self.as_shape, dtype=torch.float32, device=self.device)
            timestep = torch.zeros((self.nr_steps, self.nr_envs, self.nr_flow_samples_per_action, 1), dtype=torch.float32, device=self.device)
            initial_statistic = torch.zeros((self.nr_steps, self.nr_envs, self.nr_flow_samples_per_action), dtype=torch.float32, device=self.device)
            step_info_collection = {}

            # Acting
            with torch.inference_mode():
                for step in range(self.nr_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    if self.normalize_observation:
                        self.observation_normalizer_state = observation_normalizer.update_observation_normalizer(self.observation_normalizer_state, state, self.observation_normalizer_max_count)
                    normalized_state = self.normalize(state)
                    action, processed_action, action_info = self.sample_action(self.policy, state)
                    normalized_state_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=self.device)
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        value = self.critic(normalized_state_tensor).squeeze(-1)
                    next_state, reward, terminated, truncated, info = self.train_env.step(processed_action.cpu().numpy())
                    actual_next_state = next_state.copy()
                    for index, done in enumerate(terminated | truncated):
                        if done:
                            actual_next_state[index] = np.asarray(self.train_env.get_final_observation_at_index(info, index))
                    states[step] = normalized_state_tensor
                    next_states[step] = torch.tensor(self.normalize(actual_next_state), dtype=torch.float32, device=self.device)
                    actions[step] = action
                    rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    values[step] = value
                    terminations[step] = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                    truncations[step] = torch.tensor(truncated, dtype=torch.float32, device=self.device)
                    epsilon[step], timestep[step], initial_statistic[step] = action_info
                    for name, info_value in self.train_env.get_logging_info_dict(info).items():
                        step_info_collection.setdefault(name, []).extend(info_value)
                    state = next_state
                    global_step += self.nr_envs

                # Calculating advantages and returns
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    next_values = self.critic(next_states).squeeze(-1)
                delta = self.reward_scaling * rewards + self.gamma * (1.0 - terminations) * next_values - values
                advantages = torch.zeros_like(rewards)
                next_advantage = torch.zeros_like(rewards[-1])
                for step in range(self.nr_steps - 1, -1, -1):
                    next_advantage = advantages[step] = delta[step] + self.gamma * self.gae_lambda * (1.0 - terminations[step]) * (1.0 - truncations[step]) * next_advantage
                returns = advantages + values

            # Optimizing
            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_epsilon = epsilon.reshape((-1, self.nr_flow_samples_per_action) + self.as_shape)
            batch_timestep = timestep.reshape((-1, self.nr_flow_samples_per_action, 1))
            batch_initial_statistic = initial_statistic.reshape((-1, self.nr_flow_samples_per_action))
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(correction=0) + 1e-8)
            batch_advantages = torch.clamp(batch_advantages, -self.advantage_clamp, self.advantage_clamp)
            metrics_collection = []
            for unused_epoch in range(self.nr_epochs):
                indices = torch.randperm(self.batch_size, device=self.device)
                for start in range(0, self.batch_size, self.minibatch_size):
                    minibatch_indices = indices[start:start + self.minibatch_size]
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        current_statistic = self.compute_cfm_loss(self.policy, batch_states[minibatch_indices], batch_actions[minibatch_indices], batch_epsilon[minibatch_indices], batch_timestep[minibatch_indices])
                        initial_statistic_b = torch.minimum(batch_initial_statistic[minibatch_indices], torch.tensor(self.cfm_loss_clamp, device=self.device))
                        current_statistic = torch.minimum(current_statistic, torch.tensor(self.cfm_loss_clamp, device=self.device))
                        advantage_b = batch_advantages[minibatch_indices]
                        current_statistic = torch.where(advantage_b[..., None] < 0.0, torch.minimum(current_statistic, torch.tensor(self.cfm_loss_clamp_negative_advantages_max, device=self.device)), current_statistic)
                        unclamped_log_ratio = initial_statistic_b - current_statistic
                        clamped_log_ratio = torch.minimum(unclamped_log_ratio, torch.tensor(self.cfm_difference_clamp_max, device=self.device))
                        log_ratio = unclamped_log_ratio + (clamped_log_ratio - unclamped_log_ratio).detach()
                        ratio = torch.exp(log_ratio)
                        surrogate = -advantage_b[..., None] * ratio
                        clipped_surrogate = -advantage_b[..., None] * torch.clamp(ratio, 1.0 - self.clipping_epsilon, 1.0 + self.clipping_epsilon)
                        ppo_loss = torch.maximum(surrogate, clipped_surrogate)
                        spo_loss = -(ratio * advantage_b[..., None] - torch.abs(advantage_b[..., None]) * (ratio - 1.0) ** 2 / (2.0 * self.clipping_epsilon))
                        if self.trust_region_mode == "ppo":
                            policy_loss = ppo_loss.mean()
                        elif self.trust_region_mode == "spo":
                            policy_loss = spo_loss.mean()
                        else:
                            policy_loss = torch.where(advantage_b[..., None] > 0.0, ppo_loss, spo_loss).mean()
                        value = self.critic(batch_states[minibatch_indices]).squeeze(-1)
                        critic_loss = ((value - batch_returns[minibatch_indices]) ** 2).mean()
                        loss = policy_loss + self.critic_coef * critic_loss
                    self.policy_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    policy_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.policy.parameters() if parameter.grad is not None]))
                    critic_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.critic.parameters() if parameter.grad is not None]))
                    nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                    self.policy_optimizer.step()
                    self.critic_optimizer.step()
                    if self.anneal_learning_rate:
                        self.policy_scheduler.step()
                        self.critic_scheduler.step()
                    metrics_collection.append(torch.stack([policy_loss.detach(), critic_loss.detach(), ratio.mean().detach(), ratio.min().detach(), ratio.max().detach(), (torch.abs(ratio - 1.0) > self.clipping_epsilon).float().mean(), unclamped_log_ratio.max().detach(), (~torch.isfinite(ratio)).float().mean(), batch_actions[minibatch_indices].abs().mean(), initial_statistic_b.mean().detach(), current_statistic.mean().detach(), policy_grad_norm.detach(), critic_grad_norm.detach()]))

            self.completed_updates += 1
            if self.ema_decay > 0.0:
                if self.completed_updates == self.ema_warmup_steps:
                    self.ema_policy.load_state_dict(self.policy.state_dict())
                elif self.completed_updates > self.ema_warmup_steps:
                    with torch.no_grad():
                        for ema_parameter, policy_parameter in zip(self.ema_policy.parameters(), self.policy.parameters()):
                            ema_parameter.mul_(self.ema_decay).add_(policy_parameter, alpha=1.0 - self.ema_decay)
            metric_values = torch.stack(metrics_collection).mean(dim=0).cpu().numpy()
            metrics = {
                "loss/policy_gradient_loss": metric_values[0],
                "loss/critic_loss": metric_values[1],
                "policy_ratio/mean": metric_values[2],
                "policy_ratio/min": metric_values[3],
                "policy_ratio/max": metric_values[4],
                "policy_ratio/clip_fraction": metric_values[5],
                "policy_ratio/log_ratio_unclamped_max": metric_values[6],
                "policy_ratio/nonfinite_fraction": metric_values[7],
                "policy/latent_action_abs_mean": metric_values[8],
                "cfm/initial_loss_mean": metric_values[9],
                "cfm/current_loss_mean": metric_values[10],
                "gradients/policy_grad_norm": metric_values[11],
                "gradients/critic_grad_norm": metric_values[12],
                "lr/learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                "policy/ema_active": float(self.completed_updates > self.ema_warmup_steps),
                "v_value/explained_variance": (1.0 - torch.var(returns - values, correction=0) / (torch.var(returns, correction=0) + 1e-8)).item(),
                "time/sps": self.batch_size / (time.time() - start_time),
                "steps/nr_env_steps": global_step,
                "steps/nr_updates": self.completed_updates * self.nr_epochs * self.nr_minibatches,
            }
            for name, values_collection in step_info_collection.items():
                metric_group = "rollout" if name in ["episode_return", "episode_length"] else "env_info"
                metrics[f"{metric_group}/{name}"] = np.mean(values_collection)

            # Evaluating
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                self.set_eval_mode()
                eval_state, unused_info = self.eval_env.reset()
                completed_episodes = 0
                policy = self.ema_policy if self.completed_updates > self.ema_warmup_steps else self.policy
                while completed_episodes < self.evaluation_episodes:
                    with torch.inference_mode():
                        unused_action, eval_action, unused_action_info = self.sample_action(policy, eval_state, True)
                    eval_state, unused_reward, eval_terminated, eval_truncated, unused_info = self.eval_env.step(eval_action.cpu().numpy())
                    completed_episodes += int(np.sum(eval_terminated | eval_truncated))
                self.set_train_mode()

            # Saving
            if self.save_model and global_step >= self.total_timesteps:
                self.save()

            # Logging
            self.start_logging(global_step)
            for name, value in metrics.items():
                self.log(name, np.asarray(value), global_step)
            self.end_logging()


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
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
            "critic_state_dict": self.critic.state_dict(),
            "ema_policy_state_dict": self.ema_policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "observation_normalizer_state": self.observation_normalizer_state,
            "completed_updates": self.completed_updates,
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=self.save_path)


    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.ema_policy.load_state_dict(checkpoint["ema_policy_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.observation_normalizer_state = checkpoint["observation_normalizer_state"]
        model.completed_updates = checkpoint["completed_updates"]
        return model


    def test(self, episodes):
        self.set_eval_mode()
        policy = self.ema_policy if self.ema_decay > 0.0 and self.completed_updates > self.ema_warmup_steps else self.policy
        state, unused_info = self.eval_env.reset()
        completed_episodes = 0
        while completed_episodes < episodes:
            with torch.inference_mode():
                unused_action, processed_action, unused_action_info = self.sample_action(policy, state, True)
            state, unused_reward, terminated, truncated, unused_info = self.eval_env.step(processed_action.cpu().numpy())
            completed_episodes += int(np.sum(terminated | truncated))


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()
        self.ema_policy.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()
        self.ema_policy.eval()


    def general_properties():
        return GeneralProperties
