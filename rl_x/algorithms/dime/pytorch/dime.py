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

from rl_x.algorithms.dime.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.dime.pytorch.policy import get_policy
from rl_x.algorithms.dime.pytorch.critic import get_critic
from rl_x.algorithms.dime.pytorch.entropy_coefficient import get_entropy_coefficient
from rl_x.algorithms.dime.pytorch.replay_buffer import ReplayBuffer
from rl_x.algorithms.dime.pytorch import observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class DIME:
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
        self.actor_learning_rate = config.algorithm.actor_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.entropy_learning_rate = config.algorithm.entropy_learning_rate
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.batch_size = config.algorithm.batch_size
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.updates_per_step = config.algorithm.updates_per_step
        self.policy_delay = config.algorithm.policy_delay
        self.gamma = config.algorithm.gamma
        self.policy_tau = config.algorithm.policy_tau
        self.nr_critics = config.algorithm.nr_critics
        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.critic_entropy_coefficient = config.algorithm.critic_entropy_coefficient
        self.diffusion_steps = config.algorithm.diffusion_steps
        self.prior_std = config.algorithm.prior_std
        self.minimum_timestep = config.algorithm.minimum_timestep
        self.cosine_schedule_offset = config.algorithm.cosine_schedule_offset
        self.target_entropy_per_action_dimension = config.algorithm.target_entropy_per_action_dimension
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.action_rescaling = config.algorithm.action_rescaling
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]
        self.nr_iterations = self.total_timesteps // self.nr_envs
        self.target_entropy = self.target_entropy_per_action_dimension * self.action_dimension

        if self.total_timesteps < self.nr_envs:
            raise ValueError("Total timesteps must contain one environment step.")
        if self.buffer_size < self.learning_starts:
            raise ValueError("Replay capacity must reach the learning-start gate.")
        if self.batch_size < 1 or self.updates_per_step < 1:
            raise ValueError("Batch and update counts must be positive.")
        if self.policy_delay < 1 or self.diffusion_steps < 1:
            raise ValueError("Policy delay and diffusion steps must be positive.")
        if self.nr_critics != 2:
            raise ValueError("The reference DIME categorical update requires two critics.")
        if self.nr_atoms < 2 or self.v_max <= self.v_min:
            raise ValueError("Categorical critic support is invalid.")
        if self.prior_std <= 0.0 or self.minimum_timestep <= 0.0:
            raise ValueError("Prior standard deviation and timestep must be positive.")

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
        self.support = torch.linspace(self.v_min, self.v_max, self.nr_atoms, device=self.device)
        self.actor = get_policy(config, self.train_env, self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_actor.requires_grad_(False)
        self.critic = get_critic(config, self.train_env, self.device)
        self.entropy_coefficient = get_entropy_coefficient(config, self.device)
        self.actor.forward = torch.compile(self.actor.forward, mode=self.compile_mode)
        self.target_actor.forward = torch.compile(self.target_actor.forward, mode=self.compile_mode)
        fused = self.device.type == "cuda"
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate, betas=(self.adam_beta1, self.adam_beta2), fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, betas=(self.adam_beta1, self.adam_beta2), fused=fused)
        self.entropy_optimizer = optim.Adam(self.entropy_coefficient.parameters(), lr=self.entropy_learning_rate, fused=fused)
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)

        if self.save_model:
            os.makedirs(self.save_path)


    def normalize(self, observation):
        if self.enable_observation_normalization:
            return observation_normalizer.normalize_observation(self.observation_normalizer_state, observation, self.normalizer_epsilon)
        return observation


    def sample_action(self, actor, normalized_observation, deterministic=False):
        initial_action = self.prior_std * torch.randn(normalized_observation.shape[:-1] + self.as_shape, device=self.device)
        action = initial_action
        noise_path = torch.randn((self.diffusion_steps,) + action.shape, device=self.device)
        if deterministic:
            noise_path = torch.zeros_like(noise_path)
        base_timestep = torch.nn.functional.softplus(actor.log_timestep)
        friction = torch.nn.functional.softplus(actor.log_friction)
        log_ratio = torch.zeros(action.shape[:-1], device=self.device)
        latent_path = []
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            for step in range(self.diffusion_steps):
                timestep = torch.full(normalized_observation.shape[:-1] + (1,), step, dtype=torch.float32, device=self.device)
                reverse_time = (self.diffusion_steps - step) / self.diffusion_steps
                offset = 1.0 + self.cosine_schedule_offset
                timestep_delta = base_timestep * ((1.0 - self.minimum_timestep) * np.cos(0.5 * np.pi * (offset - reverse_time) / offset) ** 2 + self.minimum_timestep)
                variance_time = timestep_delta / friction
                transition_std = torch.sqrt(2.0 * variance_time)
                prior_score = -action / self.prior_std ** 2
                control = actor(normalized_observation, action, timestep)
                forward_mean = action + variance_time * (prior_score + control)
                next_action = forward_mean + transition_std * noise_path[step]
                backward_mean = next_action + variance_time * (-next_action / self.prior_std ** 2)
                forward_log_probability = torch.sum(-0.5 * ((next_action - forward_mean) / transition_std) ** 2 - torch.log(transition_std) - 0.5 * np.log(2.0 * np.pi), dim=-1)
                backward_log_probability = torch.sum(-0.5 * ((action - backward_mean) / transition_std) ** 2 - torch.log(transition_std) - 0.5 * np.log(2.0 * np.pi), dim=-1)
                log_ratio = log_ratio + backward_log_probability - forward_log_probability
                action = next_action
                latent_path.append(action)
        normalized_action = torch.tanh(action)
        tanh_log_determinant = torch.sum(torch.log(1.0 - normalized_action ** 2 + 1e-6), dim=-1)
        running_cost = -(log_ratio + tanh_log_determinant)
        terminal_cost = torch.sum(-0.5 * (initial_action / self.prior_std) ** 2 - np.log(self.prior_std) - 0.5 * np.log(2.0 * np.pi), dim=-1)
        return normalized_action, running_cost, torch.zeros_like(running_cost), terminal_cost, torch.stack(latent_path, dim=-2)


    def project_distribution(self, next_distribution, reward, terminated, entropy_bonus):
        target_support = torch.clamp(reward[..., None] + self.gamma * (1.0 - terminated[..., None]) * (self.support - entropy_bonus[..., None]), self.v_min, self.v_max)
        atom_delta = (self.v_max - self.v_min) / (self.nr_atoms - 1)
        position = (target_support - self.v_min) / atom_delta
        lower = torch.floor(position).long()
        upper = torch.ceil(position).long()
        lower = torch.where((upper > 0) & (lower == upper), lower - 1, lower)
        upper = torch.where((lower < self.nr_atoms - 1) & (lower == upper), upper + 1, upper)
        batch_offset = torch.arange(reward.shape[0], device=self.device)[:, None] * self.nr_atoms
        projected = torch.zeros_like(next_distribution).reshape(-1)
        projected.scatter_add_(0, (lower + batch_offset).reshape(-1), (next_distribution * (upper.float() - position)).reshape(-1))
        projected.scatter_add_(0, (upper + batch_offset).reshape(-1), (next_distribution * (position - lower.float())).reshape(-1))
        return projected.reshape(next_distribution.shape)


    def update(self, states, next_states, actions, rewards, terminations, update_count):
        with torch.no_grad():
            next_action, next_running_cost, next_stochastic_cost, next_terminal_cost, unused_path = self.sample_action(self.target_actor, next_states)
            entropy_coefficient = self.entropy_coefficient()
            entropy_bonus = entropy_coefficient * (next_running_cost + next_stochastic_cost + next_terminal_cost)
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
            current_and_next_distribution = self.critic(torch.cat([states, next_states], dim=0), torch.cat([actions, next_action], dim=0), True)
            current_distribution, next_distribution = torch.chunk(current_and_next_distribution, 2, dim=1)
            with torch.no_grad():
                target_distribution = (self.project_distribution(next_distribution[0], rewards, terminations, entropy_bonus) + self.project_distribution(next_distribution[1], rewards, terminations, entropy_bonus)) / 2.0
            cross_entropy = -torch.sum(torch.mean(torch.sum(target_distribution[None] * torch.log(current_distribution + 1e-15), dim=-1), dim=-1))
            distribution_entropy = torch.sum(torch.mean(torch.sum(current_distribution * torch.log(current_distribution + 1e-15), dim=-1), dim=-1))
            critic_loss = cross_entropy + self.critic_entropy_coefficient * distribution_entropy
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.critic.parameters() if parameter.grad is not None]))
        self.critic_optimizer.step()
        metrics = {
            "loss/critic_loss": critic_loss.detach(),
            "q/target_mean": torch.mean(torch.sum(target_distribution * self.support, dim=-1)),
            "q/current_mean": torch.mean(torch.sum(current_distribution.detach() * self.support, dim=-1)),
            "q/distribution_entropy": -torch.mean(torch.sum(current_distribution.detach() * torch.log(current_distribution.detach() + 1e-15), dim=-1)),
            "gradients/critic_grad_norm": critic_grad_norm.detach(),
        }

        if (update_count + 1) % self.policy_delay == 0:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                sampled_action, running_cost, stochastic_cost, terminal_cost, latent_path = self.sample_action(self.actor, states)
                q_distribution = self.critic(states, sampled_action, False)
                q_value = torch.mean(torch.sum(q_distribution * self.support, dim=-1), dim=0)
                path_cost = running_cost + stochastic_cost + terminal_cost
                actor_loss = torch.mean(-q_value + entropy_coefficient.detach() * path_cost)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for parameter in self.actor.parameters():
                if parameter.grad is not None:
                    parameter.grad.nan_to_num_(0.0)
            actor_grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(parameter.grad) for parameter in self.actor.parameters() if parameter.grad is not None]))
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            with torch.no_grad():
                for target_parameter, actor_parameter in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_parameter.mul_(1.0 - self.policy_tau).add_(actor_parameter, alpha=self.policy_tau)
            running_cost_mean = running_cost.mean().detach()
            entropy_loss = -self.entropy_coefficient() * (running_cost_mean - self.target_entropy)
            self.entropy_optimizer.zero_grad()
            entropy_loss.backward()
            self.entropy_optimizer.step()
            metrics.update({
                "loss/actor_loss": actor_loss.detach(),
                "entropy/running_cost": running_cost_mean,
                "entropy/stochastic_cost": stochastic_cost.mean().detach(),
                "entropy/terminal_cost": terminal_cost.mean().detach(),
                "policy/latent_abs_max": latent_path.abs().max().detach(),
                "q/policy_mean": q_value.mean().detach(),
                "loss/entropy_coefficient_loss": entropy_loss.detach(),
                "entropy/coefficient": self.entropy_coefficient().detach(),
                "gradients/actor_grad_norm": actor_grad_norm.detach(),
                "actor/update_active": torch.ones((), device=self.device),
                "entropy/target_mismatch": running_cost_mean - self.target_entropy,
            })
        else:
            metrics.update({
                "loss/actor_loss": torch.zeros((), device=self.device),
                "entropy/running_cost": torch.zeros((), device=self.device),
                "entropy/stochastic_cost": torch.zeros((), device=self.device),
                "entropy/terminal_cost": torch.zeros((), device=self.device),
                "policy/latent_abs_max": torch.zeros((), device=self.device),
                "q/policy_mean": torch.zeros((), device=self.device),
                "loss/entropy_coefficient_loss": torch.zeros((), device=self.device),
                "entropy/coefficient": self.entropy_coefficient().detach(),
                "gradients/actor_grad_norm": torch.zeros((), device=self.device),
                "actor/update_active": torch.zeros((), device=self.device),
                "entropy/target_mismatch": torch.zeros((), device=self.device),
            })
        return metrics, update_count + 1


    def train(self):
        self.set_train_mode()
        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.os_shape, self.as_shape, np.random.default_rng(self.seed))
        state, unused_info = self.train_env.reset()
        global_step = 0
        update_count = 0
        update_budget = 0.0
        metrics_collection = {}
        step_info_collection = {}
        logging_start_time = time.time()
        while global_step < self.total_timesteps:
            # Acting
            if self.enable_observation_normalization:
                self.observation_normalizer_state = observation_normalizer.update_observation_normalizer(self.observation_normalizer_state, state)
            normalized_state = self.normalize(state)
            normalized_state_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = self.sample_action(self.actor, normalized_state_tensor)
            if self.action_rescaling:
                processed_action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
            else:
                processed_action = action
            next_state, reward, terminated, truncated, info = self.train_env.step(processed_action.cpu().numpy())
            actual_next_state = next_state.copy()
            for index, done in enumerate(terminated | truncated):
                if done:
                    actual_next_state[index] = np.asarray(self.train_env.get_final_observation_at_index(info, index))
            normalized_next_state = self.normalize(actual_next_state)
            replay_buffer.add(normalized_state, normalized_next_state, action.cpu().numpy(), reward, terminated)
            for name, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(name, []).extend(info_value)
            state = next_state
            global_step += self.nr_envs

            # Updating
            if global_step >= self.learning_starts:
                update_budget += self.updates_per_step * self.nr_envs / self.batch_size
                nr_updates = int(update_budget)
                update_budget -= nr_updates
                for unused_update in range(nr_updates):
                    batch = [torch.tensor(value, dtype=torch.float32, device=self.device) for value in replay_buffer.sample(self.batch_size)]
                    metrics, update_count = self.update(*batch, update_count)
                    for name, value in metrics.items():
                        metrics_collection.setdefault(name, []).append(value.detach())

            # Evaluating
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                self.set_eval_mode()
                eval_state, unused_info = self.eval_env.reset()
                completed_episodes = 0
                while completed_episodes < self.evaluation_episodes:
                    normalized_eval_state = torch.tensor(self.normalize(eval_state), dtype=torch.float32, device=self.device)
                    with torch.inference_mode():
                        eval_action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = self.sample_action(self.actor, normalized_eval_state, True)
                    if self.action_rescaling:
                        eval_action = self.action_low + 0.5 * (eval_action + 1.0) * (self.action_high - self.action_low)
                    eval_state, unused_reward, eval_terminated, eval_truncated, unused_info = self.eval_env.step(eval_action.cpu().numpy())
                    completed_episodes += int(np.sum(eval_terminated | eval_truncated))
                self.set_train_mode()

            # Saving
            if self.save_model and global_step >= self.total_timesteps:
                self.save()

            # Logging
            if global_step % self.logging_frequency == 0:
                metrics = {name: torch.stack(values).mean().item() for name, values in metrics_collection.items()}
                for name, values in step_info_collection.items():
                    metric_group = "rollout" if name in ["episode_return", "episode_length"] else "env_info"
                    metrics[f"{metric_group}/{name}"] = np.mean(values)
                metrics["replay/fill_fraction"] = replay_buffer.size / replay_buffer.capacity
                metrics["time/sps"] = self.logging_frequency / (time.time() - logging_start_time)
                metrics["steps/nr_env_steps"] = global_step
                metrics["steps/nr_updates"] = update_count
                self.start_logging(global_step)
                for name, value in metrics.items():
                    self.log(name, value, global_step)
                self.end_logging()
                metrics_collection = {}
                step_info_collection = {}
                logging_start_time = time.time()


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
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "entropy_state_dict": self.entropy_coefficient.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
            "observation_normalizer_state": self.observation_normalizer_state,
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
        model = DIME(config, train_env, eval_env, run_path, writer)
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.target_actor.load_state_dict(checkpoint["actor_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.entropy_coefficient.load_state_dict(checkpoint["entropy_state_dict"])
        model.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        model.observation_normalizer_state = checkpoint["observation_normalizer_state"]
        return model


    def test(self, episodes):
        self.set_eval_mode()
        state, unused_info = self.eval_env.reset()
        completed_episodes = 0
        while completed_episodes < episodes:
            normalized_state = torch.tensor(self.normalize(state), dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                action, unused_running_cost, unused_stochastic_cost, unused_terminal_cost, unused_path = self.sample_action(self.actor, normalized_state, True)
            if self.action_rescaling:
                action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
            state, unused_reward, terminated, truncated, unused_info = self.eval_env.step(action.cpu().numpy())
            completed_episodes += int(np.sum(terminated | truncated))


    def set_train_mode(self):
        self.actor.train()
        self.target_actor.train()
        self.critic.train()


    def set_eval_mode(self):
        self.actor.eval()
        self.target_actor.eval()
        self.critic.eval()


    def general_properties():
        return GeneralProperties
