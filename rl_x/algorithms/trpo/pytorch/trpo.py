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

from rl_x.algorithms.trpo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.trpo.pytorch.policy import get_policy
from rl_x.algorithms.trpo.pytorch.critic import get_critic
from rl_x.algorithms.trpo.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class TRPO:
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
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.anneal_critic_learning_rate = config.algorithm.anneal_critic_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.critic_minibatch_size = config.algorithm.critic_minibatch_size
        self.nr_critic_updates = config.algorithm.nr_critic_updates
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.target_kl = config.algorithm.target_kl
        self.cg_max_steps = config.algorithm.cg_max_steps
        self.cg_damping = config.algorithm.cg_damping
        self.cg_residual_tolerance = config.algorithm.cg_residual_tolerance
        self.line_search_shrinking_factor = config.algorithm.line_search_shrinking_factor
        self.line_search_max_steps = config.algorithm.line_search_max_steps
        self.policy_subsampling_factor = config.algorithm.policy_subsampling_factor
        self.critic_max_grad_norm = config.algorithm.critic_max_grad_norm
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_critic_minibatches = self.batch_size // self.critic_minibatch_size
        self.nr_critic_optimizer_steps = self.nr_updates * self.nr_critic_updates * self.nr_critic_minibatches

        if self.batch_size % self.critic_minibatch_size != 0:
            raise ValueError("Rollout batch size must be divisible by critic minibatch size")
        if self.batch_size % self.policy_subsampling_factor != 0:
            raise ValueError("Rollout batch size must be divisible by policy subsampling factor")

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
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

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape

        self.policy = get_policy(config, self.train_env, self.device)
        self.critic = get_critic(config, self.train_env, self.device)

        fused = self.device.type == "cuda"
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate, fused=fused)

        if self.anneal_critic_learning_rate:
            self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lambda count: max(0.0, 1.0 - count / self.nr_critic_optimizer_steps))

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        @torch.jit.script
        def calculate_gae_advantages_and_returns_mixed_precision(rewards, terminations, truncations, values, next_values, gamma: float, gae_lambda: float):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                delta = rewards + gamma * next_values * (1 - terminations) - values
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros_like(rewards[0])
                for t in range(values.shape[0] - 1, -1, -1):
                    lastgaelam = advantages[t] = delta[t] + gamma * gae_lambda * (1 - terminations[t]) * (1 - truncations[t]) * lastgaelam
                returns = advantages + values
                return advantages, returns


        @torch.jit.script
        def calculate_gae_advantages_and_returns(rewards, terminations, truncations, values, next_values, gamma: float, gae_lambda: float):
            delta = rewards + gamma * next_values * (1 - terminations) - values
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[0])
            for t in range(values.shape[0] - 1, -1, -1):
                    lastgaelam = advantages[t] = delta[t] + gamma * gae_lambda * (1 - terminations[t]) * (1 - truncations[t]) * lastgaelam
            returns = advantages + values
            return advantages, returns


        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, returns):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                new_value = self.critic.get_value(states).reshape(-1)
                critic_loss = 0.5 * ((new_value - returns) ** 2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()

            critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad_norm)

            self.critic_optimizer.step()

            return critic_loss, critic_grad_norm


        self.set_train_mode()

        batch = Batch(
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            next_states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32).to(self.device),
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            values = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            terminations = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            truncations = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            log_probs = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            advantages = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            returns = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
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
                for step in range(self.nr_steps):
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        action, processed_action, log_prob = self.policy.get_action_logprob(state)
                        value = self.critic.get_value(state)
                    next_state, reward, terminated, truncated, info = self.train_env.step(processed_action.cpu().numpy())
                    done = terminated | truncated
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                    actual_next_state = next_state.clone()
                    for i, single_done in enumerate(done):
                        if single_done:
                            actual_next_state[i] = torch.tensor(np.array(self.train_env.get_final_observation_at_index(info, i), dtype=np.float32), dtype=torch.float32, device=self.device)
                            saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                            dones_this_rollout += 1
                    for key, info_value in self.train_env.get_logging_info_dict(info).items():
                        step_info_collection.setdefault(key, []).extend(info_value)

                    batch.states[step] = state
                    batch.next_states[step] = actual_next_state
                    batch.actions[step] = action
                    batch.rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                    batch.values[step] = value.reshape(-1)
                    batch.terminations[step] = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                    batch.truncations[step] = torch.tensor(truncated, dtype=torch.float32, device=self.device)
                    batch.log_probs[step] = log_prob
                    state = next_state
                    global_step += self.nr_envs
                nr_episodes += dones_this_rollout

                acting_end_time = time.time()
                time_metrics["time/acting_time"] = acting_end_time - start_time


                # Calculating advantages and returns
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    next_values = self.critic.get_value(batch.next_states).squeeze(-1)
                if self.bf16_mixed_precision_training:
                    batch.advantages, batch.returns = calculate_gae_advantages_and_returns_mixed_precision(batch.rewards, batch.terminations, batch.truncations, batch.values, next_values, self.gamma, self.gae_lambda)
                else:
                    batch.advantages, batch.returns = calculate_gae_advantages_and_returns(batch.rewards, batch.terminations, batch.truncations, batch.values, next_values, self.gamma, self.gae_lambda)

                calc_adv_return_end_time = time.time()
                time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            batch_states = batch.states.reshape((-1,) + self.os_shape)
            batch_actions = batch.actions.reshape((-1,) + self.as_shape)
            batch_advantages = batch.advantages.reshape(-1)
            batch_returns = batch.returns.reshape(-1)
            batch_values = batch.values.reshape(-1)
            batch_log_probs = batch.log_probs.reshape(-1)
            normalized_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std(correction=0) + 1e-8)
            policy_states = batch_states[::self.policy_subsampling_factor]
            policy_actions = batch_actions[::self.policy_subsampling_factor]
            policy_advantages = normalized_advantages[::self.policy_subsampling_factor]
            policy_old_log_probs = batch_log_probs[::self.policy_subsampling_factor]
            with torch.no_grad():
                old_action_mean, old_action_logstd = self.policy(policy_states)
                old_action_mean = old_action_mean.detach()
                old_action_logstd = old_action_logstd.detach()

            policy_parameters = list(self.policy.parameters())

            def policy_objective():
                new_log_prob = self.policy.get_logprob(policy_states, policy_actions)
                return (policy_advantages * torch.exp(new_log_prob - policy_old_log_probs)).mean()

            def mean_kl():
                action_mean, action_logstd = self.policy(policy_states)
                old_variance = torch.exp(2.0 * old_action_logstd)
                new_variance = torch.exp(2.0 * action_logstd)
                kl = action_logstd - old_action_logstd + (old_variance + (old_action_mean - action_mean) ** 2) / (2.0 * new_variance) - 0.5
                return kl.sum(dim=-1).mean()

            old_policy_objective = policy_objective()
            policy_gradient = torch.cat([gradient.reshape(-1) for gradient in torch.autograd.grad(old_policy_objective, policy_parameters)]).detach()

            def hessian_vector_product(vector):
                kl_gradient = torch.autograd.grad(mean_kl(), policy_parameters, create_graph=True)
                flat_kl_gradient = torch.cat([gradient.reshape(-1) for gradient in kl_gradient])
                hessian_vector = torch.autograd.grad((flat_kl_gradient * vector).sum(), policy_parameters)
                return torch.cat([value.reshape(-1) for value in hessian_vector]).detach() + self.cg_damping * vector

            solution = torch.zeros_like(policy_gradient)
            residual = policy_gradient.clone()
            direction = policy_gradient.clone()
            residual_squared = torch.dot(residual, residual)
            for _ in range(self.cg_max_steps):
                if residual_squared <= self.cg_residual_tolerance:
                    break
                hessian_direction = hessian_vector_product(direction)
                alpha = residual_squared / torch.clamp(torch.dot(direction, hessian_direction), min=1e-8)
                solution = solution + alpha * direction
                residual = residual - alpha * hessian_direction
                next_residual_squared = torch.dot(residual, residual)
                direction = residual + next_residual_squared / torch.clamp(residual_squared, min=1e-8) * direction
                residual_squared = next_residual_squared
            search_direction = solution
            search_curvature = torch.dot(search_direction, hessian_vector_product(search_direction))
            full_step = torch.sqrt(2.0 * self.target_kl / torch.clamp(search_curvature, min=1e-8)) * search_direction
            old_flat_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in policy_parameters])
            line_search_success = False
            accepted_step = self.line_search_max_steps
            new_policy_objective = old_policy_objective.detach()
            new_kl = torch.tensor(0.0, device=self.device)
            for line_search_step in range(self.line_search_max_steps):
                candidate = old_flat_parameters + self.line_search_shrinking_factor ** line_search_step * full_step
                offset = 0
                with torch.no_grad():
                    for parameter in policy_parameters:
                        parameter.copy_(candidate[offset:offset + parameter.numel()].view_as(parameter))
                        offset += parameter.numel()
                    candidate_objective = policy_objective()
                    candidate_kl = mean_kl()
                if torch.isfinite(candidate_objective) and torch.isfinite(candidate_kl) and candidate_objective > old_policy_objective and candidate_kl <= self.target_kl:
                    line_search_success = True
                    accepted_step = line_search_step
                    new_policy_objective = candidate_objective
                    new_kl = candidate_kl
                    break
            if not line_search_success:
                offset = 0
                with torch.no_grad():
                    for parameter in policy_parameters:
                        parameter.copy_(old_flat_parameters[offset:offset + parameter.numel()].view_as(parameter))
                        offset += parameter.numel()

            critic_losses = []
            critic_gradient_norms = []
            batch_indices = np.arange(self.batch_size)
            for _ in range(self.nr_critic_updates):
                self.rng.shuffle(batch_indices)
                for start in range(0, self.batch_size, self.critic_minibatch_size):
                    minibatch_indices = batch_indices[start:start + self.critic_minibatch_size]
                    critic_loss, critic_grad_norm = critic_loss_fn(batch_states[minibatch_indices], batch_returns[minibatch_indices])
                    critic_losses.append(critic_loss.item())
                    critic_gradient_norms.append(critic_grad_norm.item())
                    if self.anneal_critic_learning_rate:
                        self.critic_scheduler.step()

            y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            optimization_metrics = {
                "loss/policy_objective": new_policy_objective.item(),
                "loss/critic_loss": np.mean(critic_losses),
                "policy/kl_divergence": new_kl.item(),
                "policy/line_search_success": float(line_search_success),
                "policy/line_search_steps": accepted_step,
                "policy/line_search_fraction": self.line_search_shrinking_factor ** accepted_step if line_search_success else 0.0,
                "policy/objective_improvement": (new_policy_objective - old_policy_objective.detach()).item(),
                "policy/expected_improvement": torch.dot(policy_gradient, full_step).item(),
                "policy/std_dev": torch.exp(self.policy.policy_logstd).mean().item(),
                "conjugate_gradient/residual_norm": torch.sqrt(residual_squared).item(),
                "conjugate_gradient/search_direction_norm": torch.linalg.vector_norm(search_direction).item(),
                "gradients/policy_grad_norm": torch.linalg.vector_norm(policy_gradient).item(),
                "gradients/critic_grad_norm": np.mean(critic_gradient_norms),
                "lr/critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
                "v_value/explained_variance": explained_var,
            }

            nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                with torch.inference_mode():
                    self.set_eval_mode()
                    eval_state, _ = self.eval_env.reset()
                    eval_nr_episodes = 0
                    evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                    while True:
                        torch.compiler.cudagraph_mark_step_begin()
                        eval_state = torch.tensor(eval_state, dtype=torch.float32, device=self.device)
                        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                            eval_processed_action = self.policy.get_deterministic_action(eval_state)
                        eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.cpu().numpy())
                        eval_done = eval_terminated | eval_truncated
                        for i, single_done in enumerate(eval_done):
                            if single_done:
                                eval_nr_episodes += 1
                                evaluation_metrics["eval/episode_return"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                                evaluation_metrics["eval/episode_length"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                                if eval_nr_episodes == self.evaluation_episodes:
                                    break
                        if eval_nr_episodes == self.evaluation_episodes:
                            break
                    self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            # Also only save when there were finished episodes this update
            if self.save_model and dones_this_rollout > 0:
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
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value

            evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}

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
        torch.save({"config_algorithm": self.config.algorithm, "policy_state_dict": self.policy.state_dict(), "critic_state_dict": self.critic.state_dict(), "critic_optimizer_state_dict": self.critic_optimizer.state_dict()}, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = TRPO(config, train_env, eval_env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

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
                    state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        processed_action = self.policy.get_deterministic_action(state)
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


    def general_properties():
        return GeneralProperties
