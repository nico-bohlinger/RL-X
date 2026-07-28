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

from rl_x.algorithms.ppo_mamba2.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.ppo_mamba2.pytorch.policy import get_policy
from rl_x.algorithms.ppo_mamba2.pytorch.critic import get_critic
from rl_x.algorithms.ppo_mamba2.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class PPO_Mamba2:
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
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = self.nr_envs * self.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        self.nr_minibatch_envs = self.minibatch_size // self.nr_steps

        if self.evaluation_frequency % self.batch_size != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")
        if self.minibatch_size % self.nr_steps != 0:
            raise ValueError("Minibatch size must be a multiple of nr_steps for PPO_Mamba2.")

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
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)
        fused = self.device.type == "cuda"
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate, fused=fused)
        if self.anneal_learning_rate:
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.nr_updates)
            self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.nr_updates)
        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        @torch.jit.script
        def calculate_gae_advantages_and_returns(rewards, terminations, values, next_values, gamma: float, gae_lambda: float):
            delta = rewards + gamma * next_values * (1 - terminations) - values
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[0])
            for step in range(values.shape[0] - 1, -1, -1):
                lastgaelam = advantages[step] = delta[step] + gamma * gae_lambda * (1 - terminations[step]) * lastgaelam
            return advantages, advantages + values


        def policy_loss_fn(states, actions, log_probs, advantages, dones, init_carry):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                new_log_prob, entropy = self.policy.get_logprob_entropy(states, dones, init_carry, actions)
                logratio = new_log_prob - log_probs
                ratio = logratio.exp()
                normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                pg_loss = torch.maximum(-normalized_advantages * ratio, -normalized_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coef * entropy_loss
            self.policy_optimizer.zero_grad()
            loss.backward()
            policy_grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            with torch.no_grad():
                approx_kl_div = ((ratio - 1) - logratio).mean()
                clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean()
            return pg_loss, entropy_loss, approx_kl_div, clip_fraction, policy_grad_norm


        def critic_loss_fn(states, returns):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                critic_loss = 0.5 * (self.critic.get_value(states).squeeze(-1) - returns) ** 2
                loss = self.critic_coef * critic_loss.mean()
            self.critic_optimizer.zero_grad()
            loss.backward()
            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            return critic_loss.mean(), critic_grad_norm


        self.set_train_mode()
        carry = self.policy.initialize_carry(self.nr_envs)
        batch = Batch(
            states=torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device),
            next_states=torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device),
            actions=torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32, device=self.device),
            rewards=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            values=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            terminations=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            dones=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            log_probs=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            advantages=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            returns=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            init_policy_carry={key: torch.zeros_like(value) for key, value in carry.items()},
        )
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        state, _ = self.train_env.reset()
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
            with torch.inference_mode():
                dones_this_rollout = 0
                step_info_collection = {}
                for key in carry:
                    batch.init_policy_carry[key].copy_(carry[key])
                for step in range(self.nr_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        action, processed_action, log_prob, next_carry = self.policy.get_action_logprob(state, carry)
                        value = self.critic.get_value(state).reshape(-1)
                    next_state, reward, terminated, truncated, info = self.train_env.step(processed_action.float().cpu().numpy())
                    done = terminated | truncated
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                    actual_next_state = next_state.clone()
                    for index, single_done in enumerate(done):
                        if single_done:
                            actual_next_state[index] = torch.tensor(np.array(self.train_env.get_final_observation_at_index(info, index), dtype=np.float32), dtype=torch.float32, device=self.device)
                            saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", index))
                            dones_this_rollout += 1
                    for key, info_value in self.train_env.get_logging_info_dict(info).items():
                        step_info_collection.setdefault(key, []).extend(info_value)
                    batch.states[step] = state
                    batch.next_states[step] = actual_next_state
                    batch.actions[step] = action
                    batch.rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    batch.values[step] = value
                    batch.terminations[step] = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                    batch.dones[step] = torch.tensor(done, dtype=torch.float32, device=self.device)
                    batch.log_probs[step] = log_prob
                    state = next_state
                    carry = {key: value * (1 - batch.dones[step].reshape((self.nr_envs,) + (1,) * (value.ndim - 1))) for key, value in next_carry.items()}
                    global_step += self.nr_envs
                nr_episodes += dones_this_rollout
                acting_end_time = time.time()
                time_metrics["time/acting_time"] = acting_end_time - start_time


                # Calculating advantages and returns
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    next_values = self.critic.get_value(batch.next_states).squeeze(-1)
                batch.advantages, batch.returns = calculate_gae_advantages_and_returns(batch.rewards, batch.terminations, batch.values, next_values, self.gamma, self.gae_lambda)
                calc_adv_return_end_time = time.time()
                time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            optimization_metrics_list = []
            batch_env_indices = np.arange(self.nr_envs)
            for epoch in range(self.nr_epochs):
                self.rng.shuffle(batch_env_indices)
                for start in range(0, self.nr_envs, self.nr_minibatch_envs):
                    minibatch_env_indices = batch_env_indices[start:start + self.nr_minibatch_envs]
                    minibatch_carry = {key: value[minibatch_env_indices] for key, value in batch.init_policy_carry.items()}
                    pg_loss, entropy_loss, approx_kl_div, clip_fraction, policy_grad_norm = policy_loss_fn(batch.states[:, minibatch_env_indices], batch.actions[:, minibatch_env_indices], batch.log_probs[:, minibatch_env_indices], batch.advantages[:, minibatch_env_indices], batch.dones[:, minibatch_env_indices], minibatch_carry)
                    critic_loss, critic_grad_norm = critic_loss_fn(batch.states[:, minibatch_env_indices], batch.returns[:, minibatch_env_indices])
                    optimization_metrics_list.append({
                        "loss/policy_gradient_loss": pg_loss.item(),
                        "loss/critic_loss": critic_loss.item(),
                        "loss/entropy_loss": entropy_loss.item(),
                        "policy_ratio/approx_kl": approx_kl_div.item(),
                        "policy_ratio/clip_fraction": clip_fraction.item(),
                        "gradients/policy_grad_norm": policy_grad_norm.item(),
                        "gradients/critic_grad_norm": critic_grad_norm.item(),
                    })

            if self.anneal_learning_rate:
                self.policy_scheduler.step()
                self.critic_scheduler.step()
            optimization_metrics = {key: np.mean([metrics[key] for metrics in optimization_metrics_list]) for key in optimization_metrics_list[0]}
            optimization_metrics["lr/learning_rate"] = self.policy_optimizer.param_groups[0]["lr"]
            values = batch.values.float().cpu().numpy()
            returns = batch.returns.float().cpu().numpy()
            optimization_metrics["v_value/explained_variance"] = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)
            optimization_metrics["policy/std_dev"] = self.policy.policy_logstd.exp().mean().item()
            nr_updates += self.nr_epochs * self.nr_minibatches
            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                with torch.inference_mode():
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                    eval_carry = self.policy.initialize_carry(eval_state.shape[0])
                    eval_nr_episodes = 0
                    evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                    while True:
                        torch.compiler.cudagraph_mark_step_begin()
                        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                            eval_processed_action, next_eval_carry = self.policy.get_deterministic_action(eval_state, eval_carry)
                        eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.float().cpu().numpy())
                        eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                        eval_done = torch.tensor(eval_terminated | eval_truncated, dtype=torch.float32, device=self.device)
                        eval_carry = {key: value * (1 - eval_done.reshape((eval_state.shape[0],) + (1,) * (value.ndim - 1))) for key, value in next_eval_carry.items()}
                        for index, single_done in enumerate(eval_done):
                            if single_done:
                                eval_nr_episodes += 1
                                evaluation_metrics["eval/episode_return"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", index))
                                evaluation_metrics["eval/episode_length"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", index))
                                if eval_nr_episodes == self.evaluation_episodes:
                                    break
                        if eval_nr_episodes == self.evaluation_episodes:
                            break
                    evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                    self.set_train_mode()
            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics["time/sps"] = int(self.batch_size / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time


            # Logging
            self.start_logging(global_step)
            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes
            rollout_info_metrics = {}
            env_info_metrics = {}
            for info_name, info_values in step_info_collection.items():
                metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                mean_value = np.mean(info_values)
                if mean_value == mean_value:
                    metric_dict[f"{metric_group}/{info_name}"] = mean_value
            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(key, value, global_step)
            self.end_logging()
            logging_time_prev = time.time() - saving_end_time


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
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PPO_Mamba2(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        return model


    def test(self, episodes):
        with torch.inference_mode():
            self.set_eval_mode()
            for episode in range(episodes):
                done = False
                episode_return = 0
                state, _ = self.eval_env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                carry = self.policy.initialize_carry(state.shape[0])
                while not done:
                    torch.compiler.cudagraph_mark_step_begin()
                    with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        processed_action, next_carry = self.policy.get_deterministic_action(state, carry)
                    state, reward, terminated, truncated, info = self.eval_env.step(processed_action.float().cpu().numpy())
                    done = torch.tensor(terminated | truncated, dtype=torch.float32, device=self.device)
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    carry = {key: value * (1 - done.reshape((state.shape[0],) + (1,) * (value.ndim - 1))) for key, value in next_carry.items()}
                    episode_return += reward
                rlx_logger.info(f"Episode {episode + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.eval()


    def general_properties():
        return GeneralProperties
