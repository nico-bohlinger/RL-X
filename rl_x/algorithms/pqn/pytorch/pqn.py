import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.pqn.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.pqn.pytorch.critic import get_critic
from rl_x.algorithms.pqn.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class PQN:
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
        self.nr_epochs = config.algorithm.nr_epochs
        self.nr_minibatches = config.algorithm.nr_minibatches
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.gamma = config.algorithm.gamma
        self.q_lambda = config.algorithm.q_lambda
        self.epsilon_start = config.algorithm.epsilon_start
        self.epsilon_end = config.algorithm.epsilon_end
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = self.nr_envs * self.nr_steps
        self.nr_rollouts = self.total_timesteps // self.batch_size
        self.minibatch_size = self.batch_size // self.nr_minibatches
        self.epsilon_decay_steps = self.total_timesteps * config.algorithm.epsilon_decay_fraction

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
        self.nr_available_actions = self.train_env.get_single_action_logit_size()
        self.critic = get_critic(config, self.train_env, self.device)
        self.optimizer = optim.RAdam(self.critic.parameters(), lr=self.learning_rate)
        if self.anneal_learning_rate:
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda count: 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_rollouts)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf


    def train(self):
        @torch.jit.script
        def calculate_q_targets(rewards, terminations, next_values, gamma: float, q_lambda: float):
            q_targets = torch.zeros_like(rewards)
            q_targets[-1] = rewards[-1] + gamma * next_values[-1] * (1.0 - terminations[-1])
            for step in range(rewards.shape[0] - 2, -1, -1):
                mixed_bootstrap = q_lambda * q_targets[step + 1] + (1 - q_lambda) * next_values[step]
                q_targets[step] = rewards[step] + gamma * mixed_bootstrap * (1.0 - terminations[step])
            return q_targets


        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, actions, q_targets):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                q = self.critic(states).gather(1, actions.reshape(-1, 1)).squeeze(1)
                q_loss = 0.5 * ((q - q_targets) ** 2).mean()

            self.optimizer.zero_grad()
            q_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            return q_loss, q.mean(), critic_grad_norm


        self.set_train_mode()

        batch = Batch(
            states=torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device),
            next_states=torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32, device=self.device),
            actions=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.int64, device=self.device),
            rewards=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            terminations=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
            q_targets=torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32, device=self.device),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        state, _ = self.train_env.reset()
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
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * min(1.0, global_step / self.epsilon_decay_steps)
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    greedy_action = self.critic(state_tensor).argmax(dim=-1).cpu().numpy()
                random_action = self.rng.integers(self.nr_available_actions, size=self.nr_envs)
                action = np.where(self.rng.random(self.nr_envs) < epsilon, random_action, greedy_action)

                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.train_env.get_final_observation_at_index(info, i)
                        saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state_tensor
                batch.next_states[step] = torch.tensor(actual_next_state, dtype=torch.float32, device=self.device)
                batch.actions[step] = torch.tensor(action, dtype=torch.int64, device=self.device)
                batch.rewards[step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
                batch.terminations[step] = torch.tensor(terminated, dtype=torch.float32, device=self.device)
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating Q targets
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                next_values = self.critic(batch.next_states.reshape((-1,) + self.os_shape)).max(dim=-1).values.reshape(self.nr_steps, self.nr_envs)
                batch.q_targets = calculate_q_targets(batch.rewards, batch.terminations, next_values, self.gamma, self.q_lambda)

            calc_q_target_end_time = time.time()
            time_metrics["time/calc_q_target_time"] = calc_q_target_end_time - acting_end_time


            # Optimizing
            batch_states = batch.states.reshape((-1,) + self.os_shape)
            batch_actions = batch.actions.reshape(-1)
            batch_q_targets = batch.q_targets.reshape(-1)
            optimization_metrics_collection = {}
            for epoch in range(self.nr_epochs):
                batch_indices = torch.randperm(self.batch_size, device=self.device)
                for start in range(0, self.batch_size, self.minibatch_size):
                    minibatch_indices = batch_indices[start:start + self.minibatch_size]
                    q_loss, q, critic_grad_norm = critic_loss_fn(batch_states[minibatch_indices], batch_actions[minibatch_indices], batch_q_targets[minibatch_indices])
                    optimization_metrics = {
                        "loss/q_loss": q_loss.item(),
                        "q_value/q_value": q.item(),
                        "gradients/critic_grad_norm": critic_grad_norm.item(),
                    }
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    if self.anneal_learning_rate:
                        self.scheduler.step()
            optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
            optimization_metrics["lr/learning_rate"] = self.optimizer.param_groups[0]["lr"]
            optimization_metrics["epsilon/epsilon"] = epsilon
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_q_target_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        eval_action = self.critic(torch.tensor(eval_state, dtype=torch.float32, device=self.device)).argmax(dim=-1).cpu().numpy()
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_action)
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
        torch.save({
            "config_algorithm": self.config.algorithm,
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PQN(config, train_env, eval_env, run_path, writer)
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
                    action = self.critic(torch.tensor(state, dtype=torch.float32, device=self.device)).argmax(dim=-1).cpu().numpy()
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.critic.train()


    def set_eval_mode(self):
        self.critic.eval()


    def general_properties():
        return GeneralProperties
