import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.bro.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.bro.pytorch.policy import get_policy
from rl_x.algorithms.bro.pytorch.critic import get_critic
from rl_x.algorithms.bro.pytorch.coefficients import Adjustment, EntropyCoefficient
from rl_x.algorithms.bro.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class BRO:
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
        self.entropy_coefficient_learning_rate = config.algorithm.entropy_coefficient_learning_rate
        self.adjustment_learning_rate = config.algorithm.adjustment_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.gamma = config.algorithm.gamma
        self.tau = config.algorithm.tau
        self.distributional = config.algorithm.distributional
        self.nr_quantiles = config.algorithm.nr_quantiles
        self.pessimism = config.algorithm.pessimism
        self.kl_target = config.algorithm.kl_target
        self.std_multiplier = config.algorithm.std_multiplier
        self.log_value_min = config.algorithm.log_value_min
        self.log_value_max = config.algorithm.log_value_max
        self.use_optimistic_exploration = config.algorithm.use_optimistic_exploration
        self.first_reset_step = config.algorithm.first_reset_step
        self.reset_interval = config.algorithm.reset_interval
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.action_dim = self.train_env.single_action_space.shape[0]
        self.target_entropy = -self.action_dim / 2 if config.algorithm.target_entropy == "auto" else float(config.algorithm.target_entropy)
        self.quantile_taus = ((torch.arange(self.nr_quantiles + 1, dtype=torch.float32)[:-1] + torch.arange(self.nr_quantiles + 1, dtype=torch.float32)[1:]) / (2.0 * self.nr_quantiles)).view(1, self.nr_quantiles, 1)

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
        torch.backends.cudnn.deterministic = True

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        self.build_networks()

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def build_networks(self):
        torch.manual_seed(self.seed)
        self.policy, self.optimistic_policy = get_policy(self.config, self.train_env, self.device)
        self.critic = get_critic(self.config, self.train_env, self.device)
        self.target_critic = get_critic(self.config, self.train_env, self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.entropy_coefficient = EntropyCoefficient(self.config.algorithm.init_entropy_coefficient).to(self.device)
        self.optimism = Adjustment(self.config.algorithm.init_optimism, self.log_value_min, self.log_value_max).to(self.device)
        self.regularizer = Adjustment(self.config.algorithm.init_regularizer, self.log_value_min, self.log_value_max).to(self.device)

        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.policy_learning_rate, weight_decay=1e-4, fused=fused)
        self.optimistic_policy_optimizer = optim.AdamW(self.optimistic_policy.parameters(), lr=self.policy_learning_rate, weight_decay=1e-4, fused=fused)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=1e-4, fused=fused)
        self.entropy_optimizer = optim.Adam(self.entropy_coefficient.parameters(), lr=self.entropy_coefficient_learning_rate, betas=(0.5, 0.999), fused=fused)
        self.optimism_optimizer = optim.Adam(self.optimism.parameters(), lr=self.adjustment_learning_rate, betas=(0.5, 0.999), fused=fused)
        self.regularizer_optimizer = optim.Adam(self.regularizer.parameters(), lr=self.adjustment_learning_rate, betas=(0.5, 0.999), fused=fused)


    def train(self):
        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, terminations):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(next_states)
                    next_q1, next_q2 = self.target_critic(next_states, next_actions)
                    next_q = (next_q1 + next_q2) / 2.0 - self.pessimism * torch.abs(next_q1 - next_q2) / 2.0
                    if self.distributional:
                        target_q = rewards[:, None, None] + self.gamma * (1.0 - terminations[:, None, None]) * next_q[:, None, :]
                        target_q = target_q - self.gamma * self.entropy_coefficient() * (1.0 - terminations[:, None, None]) * next_log_probs[:, None, None]
                    else:
                        target_q = rewards + self.gamma * (1.0 - terminations) * (next_q - self.entropy_coefficient() * next_log_probs)
                q1, q2 = self.critic(states, actions)
                if self.distributional:
                    quantile_taus = self.quantile_taus.to(device=self.device, dtype=q1.dtype)
                    td1 = target_q - q1[..., None]
                    td2 = target_q - q2[..., None]
                    huber1 = torch.where(torch.abs(td1) <= 1.0, 0.5 * td1 ** 2, torch.abs(td1) - 0.5)
                    huber2 = torch.where(torch.abs(td2) <= 1.0, 0.5 * td2 ** 2, torch.abs(td2) - 0.5)
                    critic_loss = (torch.abs(quantile_taus - (td1 < 0).to(td1.dtype).detach()) * huber1).sum(dim=1).mean()
                    critic_loss = critic_loss + (torch.abs(quantile_taus - (td2 < 0).to(td2.dtype).detach()) * huber2).sum(dim=1).mean()
                else:
                    critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            return critic_loss, q1.mean(), q2.mean()


        @torch.compile(mode=self.compile_mode)
        def policy_loss_fn(states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                actions, _, log_probs = self.policy.get_action(states)
                q1, q2 = self.critic(states, actions)
                q = (q1 + q2) / 2.0 - self.pessimism * torch.abs(q1 - q2) / 2.0
                if self.distributional:
                    q = q.mean(dim=-1)
                entropy = -log_probs.mean()
                policy_loss = (self.entropy_coefficient().detach() * log_probs - q).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            return policy_loss, entropy.detach(), q.mean()


        @torch.compile(mode=self.compile_mode)
        def optimistic_policy_loss_fn(states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    pessimistic_mean, pessimistic_std = self.policy(states)
                actions, _, _, optimistic_mean, optimistic_std = self.optimistic_policy.get_action(states, pessimistic_mean, pessimistic_std, self.std_multiplier)
                q1, q2 = self.critic(states, actions)
                q_upper_bound = (q1 + q2) / 2.0 + self.optimism().detach() * torch.abs(q1 - q2) / 2.0
                if self.distributional:
                    q_upper_bound = q_upper_bound.mean(dim=-1)
                effective_optimistic_std = optimistic_std / self.std_multiplier
                kl = (torch.log(pessimistic_std / effective_optimistic_std) + (effective_optimistic_std ** 2 + (optimistic_mean - pessimistic_mean) ** 2) / (2.0 * pessimistic_std ** 2) - 0.5).sum(dim=-1)
                optimistic_policy_loss = -q_upper_bound.mean() + self.regularizer().detach() * kl.mean()

            self.optimistic_policy_optimizer.zero_grad()
            optimistic_policy_loss.backward()
            self.optimistic_policy_optimizer.step()
            return optimistic_policy_loss, kl.mean().detach()


        @torch.compile(mode=self.compile_mode)
        def coefficient_loss_fn(entropy, empirical_kl):
            entropy_coefficient = self.entropy_coefficient()
            entropy_coefficient_loss = entropy_coefficient * (entropy - self.target_entropy)
            self.entropy_optimizer.zero_grad()
            entropy_coefficient_loss.backward()
            self.entropy_optimizer.step()

            optimism = self.optimism()
            optimism_loss = (optimism - self.pessimism) * (empirical_kl - self.kl_target)
            self.optimism_optimizer.zero_grad()
            optimism_loss.backward()
            self.optimism_optimizer.step()

            regularizer = self.regularizer()
            regularizer_loss = -regularizer * (empirical_kl - self.kl_target)
            self.regularizer_optimizer.zero_grad()
            regularizer_loss.backward()
            self.regularizer_optimizer.step()
            return entropy_coefficient_loss, entropy_coefficient.detach(), optimism.detach(), regularizer.detach()


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.rng, self.device)
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        reset_steps = set()
        if self.reset_interval > 0:
            reset_steps.add(self.first_reset_step)
            reset_step = self.reset_interval
            while reset_step < self.total_timesteps:
                reset_steps.add(reset_step)
                reset_step += self.reset_interval
        reset_steps = deque(sorted(reset_steps))

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
            if global_step < self.learning_starts:
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                    if self.use_optimistic_exploration:
                        pessimistic_mean, pessimistic_std = self.policy(state_tensor)
                        action, processed_action, _, _, _ = self.optimistic_policy.get_action(state_tensor, pessimistic_mean, pessimistic_std, self.std_multiplier)
                    else:
                        action, processed_action, _ = self.policy.get_action(state_tensor)
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
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_reset = bool(reset_steps) and global_step >= reset_steps[0]
            should_optimize = should_learning_start
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Resetting
            if should_reset:
                rlx_logger.info(f"Resetting all networks at step {global_step}")
                while reset_steps and global_step >= reset_steps[0]:
                    reset_steps.popleft()
                self.build_networks()
                self.set_train_mode()


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)
                    critic_loss, q1_mean, q2_mean = critic_loss_fn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations)
                    with torch.no_grad():
                        for parameter, target_parameter in zip(self.critic.parameters(), self.target_critic.parameters()):
                            target_parameter.data.mul_(1.0 - self.tau).add_(parameter.data, alpha=self.tau)
                    policy_loss, entropy, policy_q_mean = policy_loss_fn(batch_states)
                    optimistic_policy_loss, kl_mean = optimistic_policy_loss_fn(batch_states)
                    empirical_kl = kl_mean / self.action_dim
                    entropy_coefficient_loss, entropy_coefficient, optimism, regularizer = coefficient_loss_fn(entropy, empirical_kl)

                    optimization_metrics = {
                        "entropy/entropy": entropy.item(),
                        "entropy/entropy_coefficient": entropy_coefficient.item(),
                        "kl/empirical_kl": empirical_kl.item(),
                        "loss/critic_loss": critic_loss.item(),
                        "loss/policy_loss": policy_loss.item(),
                        "loss/optimistic_policy_loss": optimistic_policy_loss.item(),
                        "loss/entropy_coefficient_loss": entropy_coefficient_loss.item(),
                        "optimism/value": optimism.item(),
                        "regularizer/value": regularizer.item(),
                        "q/q1_mean": q1_mean.item(),
                        "q/q2_mean": q2_mean.item(),
                        "q/policy_q_mean": policy_q_mean.item(),
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
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        eval_processed_action = self.policy.get_deterministic_action(torch.tensor(eval_state, dtype=torch.float32, device=self.device))
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
            "optimistic_policy_state_dict": self.optimistic_policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "entropy_coefficient_state_dict": self.entropy_coefficient.state_dict(),
            "optimism_state_dict": self.optimism.state_dict(),
            "regularizer_state_dict": self.regularizer.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "optimistic_policy_optimizer_state_dict": self.optimistic_policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
            "optimism_optimizer_state_dict": self.optimism_optimizer.state_dict(),
            "regularizer_optimizer_state_dict": self.regularizer_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = BRO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.optimistic_policy.load_state_dict(checkpoint["optimistic_policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])
        model.entropy_coefficient.load_state_dict(checkpoint["entropy_coefficient_state_dict"])
        model.optimism.load_state_dict(checkpoint["optimism_state_dict"])
        model.regularizer.load_state_dict(checkpoint["regularizer_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.optimistic_policy_optimizer.load_state_dict(checkpoint["optimistic_policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        model.optimism_optimizer.load_state_dict(checkpoint["optimism_optimizer_state_dict"])
        model.regularizer_optimizer.load_state_dict(checkpoint["regularizer_optimizer_state_dict"])
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
                    processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32, device=self.device))
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.policy.train()
        self.optimistic_policy.train()
        self.critic.train()
        self.target_critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.optimistic_policy.eval()
        self.critic.eval()
        self.target_critic.eval()


    def general_properties():
        return GeneralProperties
