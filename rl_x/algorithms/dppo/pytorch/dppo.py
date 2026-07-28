import os
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.dppo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.dppo.pytorch.policy import get_policy
from rl_x.algorithms.dppo.pytorch.critic import get_critic
from rl_x.algorithms.dppo.pytorch import observation_normalizer
from rl_x.algorithms.dppo.pytorch import reward_normalizer

rlx_logger = logging.getLogger("rl_x")


class DPPO:
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
        self.policy_learning_rate = config.algorithm.policy_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clipping_epsilon = config.algorithm.clipping_epsilon
        self.clipping_epsilon_base = config.algorithm.clipping_epsilon_base
        self.clipping_epsilon_rate = config.algorithm.clipping_epsilon_rate
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.target_kl = config.algorithm.target_kl
        self.reward_scaling = config.algorithm.reward_scaling
        self.normalize_reward = config.algorithm.normalize_reward
        self.reward_clip = config.algorithm.reward_clip
        self.normalize_observation = config.algorithm.normalize_observation
        self.action_rescaling = config.algorithm.action_rescaling
        self.diffusion_steps = config.algorithm.diffusion_steps
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.policy_hidden_dims = tuple(config.algorithm.policy_hidden_dims)
        self.critic_hidden_dims = tuple(config.algorithm.critic_hidden_dims)
        self.denoising_std = config.algorithm.denoising_std
        self.denoising_discount = config.algorithm.denoising_discount
        self.denoised_clip_value = config.algorithm.denoised_clip_value
        self.noise_clip_value = config.algorithm.noise_clip_value
        self.log_probability_min = config.algorithm.log_probability_min
        self.log_probability_max = config.algorithm.log_probability_max
        self.advantage_quantile_min = config.algorithm.advantage_quantile_min
        self.advantage_quantile_max = config.algorithm.advantage_quantile_max
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.batch_size = self.nr_envs * self.nr_steps
        self.optimization_batch_size = self.batch_size * self.diffusion_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.optimization_batch_size // self.minibatch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.optimization_batch_size % self.minibatch_size != 0:
            raise ValueError("The denoising-MDP batch must be divisible by the minibatch size.")
        if self.diffusion_steps < 2:
            raise ValueError("DPPO requires at least two diffusion steps.")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be positive and divisible by two.")
        if self.denoising_std <= 0.0:
            raise ValueError("Denoising standard deviation must be positive.")
        if len(self.policy_hidden_dims) % 2 != 1 or len(self.critic_hidden_dims) % 2 != 1:
            raise ValueError("Residual networks require an odd number of hidden dimensions.")
        if not 0.0 <= self.advantage_quantile_min < self.advantage_quantile_max <= 1.0:
            raise ValueError("Advantage quantiles must be ordered inside [0, 1].")

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
        self.action_low = torch.tensor(self.train_env.single_action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(self.train_env.single_action_space.high, dtype=torch.float32, device=self.device)
        cosine_positions = torch.linspace(0.0, self.diffusion_steps + 1, self.diffusion_steps + 1, device=self.device)
        alpha_cumulative = torch.cos((cosine_positions / (self.diffusion_steps + 1) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alpha_cumulative = alpha_cumulative / alpha_cumulative[0]
        self.betas = torch.clamp(1.0 - alpha_cumulative[1:] / alpha_cumulative[:-1], 0.0, 0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumulative = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumulative_previous = torch.cat([torch.ones(1, device=self.device), self.alphas_cumulative[:-1]])
        self.sqrt_reciprocal_alphas_cumulative = torch.sqrt(1.0 / self.alphas_cumulative)
        self.sqrt_reciprocal_minus_one_alphas_cumulative = torch.sqrt(1.0 / self.alphas_cumulative - 1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumulative_previous) / (1.0 - self.alphas_cumulative)
        self.posterior_mean_coefficient_1 = self.betas * torch.sqrt(self.alphas_cumulative_previous) / (1.0 - self.alphas_cumulative)
        self.posterior_mean_coefficient_2 = (1.0 - self.alphas_cumulative_previous) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumulative)

        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        self.policy.forward = torch.compile(self.policy.forward, mode=self.compile_mode)
        self.critic.forward = torch.compile(self.critic.forward, mode=self.compile_mode)
        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_learning_rate, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, fused=fused)
        if self.anneal_learning_rate:
            steps_per_update = self.nr_minibatches * self.nr_epochs
            self.policy_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lambda count: max(0.0, 1.0 - (count // steps_per_update) / self.nr_updates))
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)
        self.reward_normalizer_state = reward_normalizer.init_reward_normalizer_state(self.nr_envs)

        if self.save_model:
            os.makedirs(self.save_path)


    def normalize(self, observation):
        if self.normalize_observation:
            return observation_normalizer.normalize_observation(self.observation_normalizer_state, observation)
        return observation


    def compute_transition_log_likelihood(self, normalized_observation, path_current, path_next, denoising_index):
        if path_current.ndim == normalized_observation.ndim + 1:
            normalized_observation = torch.broadcast_to(normalized_observation[..., None, :], path_current.shape[:-1] + (normalized_observation.shape[-1],))
        diffusion_timestep = self.diffusion_steps - denoising_index - 1
        timestep = torch.broadcast_to(diffusion_timestep[..., None].float(), path_current.shape[:-1] + (1,))
        predicted_noise = self.policy(normalized_observation, path_current, timestep)
        reconstructed_action = self.sqrt_reciprocal_alphas_cumulative[diffusion_timestep][..., None] * path_current - self.sqrt_reciprocal_minus_one_alphas_cumulative[diffusion_timestep][..., None] * predicted_noise
        reconstructed_action = torch.clamp(reconstructed_action, -self.denoised_clip_value, self.denoised_clip_value)
        transition_mean = self.posterior_mean_coefficient_1[diffusion_timestep][..., None] * reconstructed_action + self.posterior_mean_coefficient_2[diffusion_timestep][..., None] * path_current
        transition_std = torch.maximum(torch.sqrt(self.posterior_variance[diffusion_timestep])[..., None], torch.tensor(self.denoising_std, device=self.device))
        standardized_noise = (path_next - transition_mean) / transition_std
        return -0.5 * standardized_noise ** 2 - 0.5 * torch.log(2.0 * torch.pi * transition_std ** 2)


    def sample_action(self, observation, deterministic=False):
        normalized_observation = torch.tensor(self.normalize(observation), dtype=torch.float32, device=self.device)
        action = torch.randn(observation.shape[:-1] + self.as_shape, device=self.device)
        noise_path = torch.randn((self.diffusion_steps,) + action.shape, device=self.device)
        path = []
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            for denoising_index, diffusion_timestep in enumerate(range(self.diffusion_steps - 1, -1, -1)):
                path.append(action)
                timestep = torch.full(observation.shape[:-1] + (1,), diffusion_timestep, dtype=torch.float32, device=self.device)
                predicted_noise = self.policy(normalized_observation, action, timestep)
                reconstructed_action = self.sqrt_reciprocal_alphas_cumulative[diffusion_timestep] * action - self.sqrt_reciprocal_minus_one_alphas_cumulative[diffusion_timestep] * predicted_noise
                reconstructed_action = torch.clamp(reconstructed_action, -self.denoised_clip_value, self.denoised_clip_value)
                transition_mean = self.posterior_mean_coefficient_1[diffusion_timestep] * reconstructed_action + self.posterior_mean_coefficient_2[diffusion_timestep] * action
                transition_std = torch.maximum(torch.sqrt(self.posterior_variance[diffusion_timestep]), torch.tensor(1e-3 if deterministic else self.denoising_std, device=self.device))
                if deterministic and diffusion_timestep == 0:
                    transition_std = torch.zeros_like(transition_std)
                action = transition_mean + transition_std * torch.clamp(noise_path[denoising_index], -self.noise_clip_value, self.noise_clip_value)
            full_path = torch.cat([torch.stack(path, dim=-2), action[..., None, :]], dim=-2)
            behavior_log_likelihood = self.compute_transition_log_likelihood(normalized_observation, full_path[..., :-1, :], full_path[..., 1:, :], torch.arange(self.diffusion_steps, device=self.device))
        if self.action_rescaling:
            processed_action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
        else:
            processed_action = action
        return action, processed_action, full_path, behavior_log_likelihood


    def train(self):
        self.set_train_mode()
        state, unused_info = self.train_env.reset()
        global_step = 0
        completed_updates = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device)
            next_states = torch.zeros_like(states)
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32, device=self.device)
            full_paths = torch.zeros((self.nr_steps, self.nr_envs, self.diffusion_steps + 1) + self.as_shape, dtype=torch.float32, device=self.device)
            behavior_log_likelihoods = torch.zeros((self.nr_steps, self.nr_envs, self.diffusion_steps, self.action_dimension), dtype=torch.float32, device=self.device)
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device)
            values = torch.zeros_like(rewards)
            terminations = torch.zeros_like(rewards)
            truncations = torch.zeros_like(rewards)
            step_info_collection = {}

            # Acting
            with torch.inference_mode():
                for step in range(self.nr_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    if self.normalize_observation:
                        self.observation_normalizer_state = observation_normalizer.update_observation_normalizer(self.observation_normalizer_state, state)
                    normalized_state = self.normalize(state)
                    action, processed_action, full_path, behavior_log_likelihood = self.sample_action(state)
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
                    full_paths[step] = full_path
                    behavior_log_likelihoods[step] = behavior_log_likelihood
                    rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    values[step] = value
                    terminations[step] = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                    truncations[step] = torch.tensor(truncated, dtype=torch.float32, device=self.device)
                    for name, info_value in self.train_env.get_logging_info_dict(info).items():
                        step_info_collection.setdefault(name, []).extend(info_value)
                    state = next_state
                    global_step += self.nr_envs

                # Calculating advantages and returns
                normalized_rewards = rewards.cpu().numpy()
                if self.normalize_reward:
                    self.reward_normalizer_state, normalized_rewards = reward_normalizer.normalize_reward(self.reward_normalizer_state, normalized_rewards, terminations.cpu().numpy(), truncations.cpu().numpy(), self.gamma, self.reward_clip)
                normalized_rewards = torch.tensor(normalized_rewards, dtype=torch.float32, device=self.device)
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    next_values = self.critic(next_states).squeeze(-1)
                delta = self.reward_scaling * normalized_rewards + self.gamma * (1.0 - terminations) * next_values - values
                advantages = torch.zeros_like(rewards)
                next_advantage = torch.zeros_like(rewards[-1])
                for step in range(self.nr_steps - 1, -1, -1):
                    next_advantage = advantages[step] = delta[step] + self.gamma * self.gae_lambda * (1.0 - terminations[step]) * (1.0 - truncations[step]) * next_advantage
                returns = advantages + values

            # Optimizing
            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_full_paths = full_paths.reshape((-1, self.diffusion_steps + 1) + self.as_shape)
            batch_behavior_log_likelihoods = behavior_log_likelihoods.reshape((-1, self.diffusion_steps, self.action_dimension))
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            metrics_collection = []
            update_active = True
            for unused_epoch in range(self.nr_epochs):
                indices = torch.randperm(self.optimization_batch_size, device=self.device)
                for start in range(0, self.optimization_batch_size, self.minibatch_size):
                    minibatch_indices = indices[start:start + self.minibatch_size]
                    transition_indices = minibatch_indices // self.diffusion_steps
                    denoising_indices = minibatch_indices % self.diffusion_steps
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        current_log_likelihood = self.compute_transition_log_likelihood(batch_states[transition_indices], batch_full_paths[transition_indices, denoising_indices], batch_full_paths[transition_indices, denoising_indices + 1], denoising_indices)
                        current_log_likelihood = torch.clamp(current_log_likelihood, self.log_probability_min, self.log_probability_max)
                        behavior_log_likelihood = torch.clamp(batch_behavior_log_likelihoods[transition_indices, denoising_indices], self.log_probability_min, self.log_probability_max)
                        log_ratio = torch.mean(current_log_likelihood - behavior_log_likelihood, dim=-1)
                        ratio = torch.exp(log_ratio)
                        advantage = batch_advantages[transition_indices]
                        advantage = (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
                        advantage = torch.clamp(advantage, torch.quantile(advantage, self.advantage_quantile_min), torch.quantile(advantage, self.advantage_quantile_max))
                        advantage = advantage * self.denoising_discount ** (self.diffusion_steps - denoising_indices - 1)
                        denoising_fraction = denoising_indices / (self.diffusion_steps - 1)
                        clipping_epsilon = self.clipping_epsilon_base + (self.clipping_epsilon - self.clipping_epsilon_base) * (torch.exp(self.clipping_epsilon_rate * denoising_fraction) - 1.0) / (np.exp(self.clipping_epsilon_rate) - 1.0)
                        surrogate = ratio * advantage
                        clipped_surrogate = torch.clamp(ratio, 1.0 - clipping_epsilon, 1.0 + clipping_epsilon) * advantage
                        policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
                        value = self.critic(batch_states[transition_indices]).squeeze(-1)
                        critic_loss = 0.5 * ((value - batch_returns[transition_indices]) ** 2).mean()
                        loss = policy_loss + self.critic_coef * critic_loss
                    self.policy_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    policy_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.policy.parameters() if parameter.grad is not None]))
                    critic_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.critic.parameters() if parameter.grad is not None]))
                    if self.max_grad_norm != -1.0:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()
                    self.critic_optimizer.step()
                    if self.anneal_learning_rate:
                        self.policy_scheduler.step()
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    metrics_collection.append(torch.stack([policy_loss.detach(), critic_loss.detach(), ratio.mean().detach(), ratio.min().detach(), ratio.max().detach(), (torch.abs(ratio - 1.0) > clipping_epsilon).float().mean(), approx_kl.detach(), torch.abs(log_ratio).max().detach(), denoising_indices.float().mean(), clipping_epsilon.mean(), batch_actions[transition_indices].abs().mean(), policy_grad_norm.detach(), critic_grad_norm.detach()]))
                    if self.target_kl is not None and approx_kl > self.target_kl:
                        update_active = False
                        break
                if not update_active:
                    break

            completed_updates += 1
            metric_values = torch.stack(metrics_collection).mean(dim=0).cpu().numpy()
            metrics = {
                "loss/policy_gradient_loss": metric_values[0],
                "loss/critic_loss": metric_values[1],
                "policy_ratio/mean": metric_values[2],
                "policy_ratio/min": metric_values[3],
                "policy_ratio/max": metric_values[4],
                "policy_ratio/clip_fraction": metric_values[5],
                "policy_ratio/approx_kl": metric_values[6],
                "policy_ratio/log_ratio_abs_max": metric_values[7],
                "diffusion/denoising_index_mean": metric_values[8],
                "diffusion/clipping_epsilon_mean": metric_values[9],
                "policy/latent_action_abs_mean": metric_values[10],
                "gradients/policy_grad_norm": metric_values[11],
                "gradients/critic_grad_norm": metric_values[12],
                "optimization/update_active": float(update_active),
                "lr/policy_learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                "lr/critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
                "v_value/explained_variance": (1.0 - torch.var(returns - values, correction=0) / (torch.var(returns, correction=0) + 1e-8)).item(),
                "time/sps": self.batch_size / (time.time() - start_time),
                "steps/nr_env_steps": global_step,
                "steps/nr_updates": completed_updates * self.nr_epochs * self.nr_minibatches,
            }
            for name, values_collection in step_info_collection.items():
                metric_group = "rollout" if name in ["episode_return", "episode_length"] else "env_info"
                metrics[f"{metric_group}/{name}"] = np.mean(values_collection)

            # Evaluating
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                self.set_eval_mode()
                eval_state, unused_info = self.eval_env.reset()
                completed_episodes = 0
                while completed_episodes < self.evaluation_episodes:
                    with torch.inference_mode():
                        unused_action, eval_action, unused_path, unused_log_likelihood = self.sample_action(eval_state, True)
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
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "observation_normalizer_state": self.observation_normalizer_state,
            "reward_normalizer_state": self.reward_normalizer_state,
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
        model = DPPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.observation_normalizer_state = checkpoint["observation_normalizer_state"]
        model.reward_normalizer_state = checkpoint["reward_normalizer_state"]
        return model


    def test(self, episodes):
        self.set_eval_mode()
        state, unused_info = self.eval_env.reset()
        completed_episodes = 0
        while completed_episodes < episodes:
            with torch.inference_mode():
                unused_action, processed_action, unused_path, unused_log_likelihood = self.sample_action(state, True)
            state, unused_reward, terminated, truncated, unused_info = self.eval_env.step(processed_action.cpu().numpy())
            completed_episodes += int(np.sum(terminated | truncated))


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()


    def general_properties():
        return GeneralProperties
