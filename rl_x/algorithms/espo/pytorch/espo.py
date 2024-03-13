import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from rl_x.algorithms.espo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.espo.pytorch.policy import get_policy
from rl_x.algorithms.espo.pytorch.critic import get_critic
from rl_x.algorithms.espo.pytorch.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class ESPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.max_epochs = config.algorithm.max_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_ratio_delta = config.algorithm.max_ratio_delta
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        if config.algorithm.delta_calc_operator == "mean":
            self.delta_calc_operator = torch.mean
        elif config.algorithm.delta_calc_operator == "median":
            self.delta_calc_operator = torch.median
        else:
            raise ValueError("Unknown delta_calc_operator")

        if config.algorithm.device == "gpu" and torch.cuda.is_available():
            device_name = "cuda"
        elif config.algorithm.device == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device_name = "mps"
        else:
            device_name = "cpu"
        self.device = torch.device(device_name)
        rlx_logger.info(f"Using device: {self.device}")

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape

        self.policy = torch.compile(get_policy(config, env, self.device).to(self.device), mode="default")
        self.critic = torch.compile(get_critic(config, env).to(self.device), mode="default")
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        @torch.jit.script
        def calculate_gae_advantages_and_returns(rewards, terminations, values, next_values, gamma: float, gae_lambda: float):
            delta = rewards + gamma * next_values * (1 - terminations) - values
            advantages = torch.zeros_like(rewards)
            lastgaelam = torch.zeros_like(rewards[0])
            for t in range(values.shape[0] - 2, -1, -1):
                lastgaelam = advantages[t] = delta[t] + gamma * gae_lambda * (1 - terminations[t]) * lastgaelam
            returns = advantages + values
            return advantages, returns
        

        @torch.compile(mode="default")
        def policy_loss_fn(policy, states, actions, log_probs, advantages):
            new_log_prob, entropy = policy.get_logprob_entropy(states, actions)
            logratio = new_log_prob - log_probs
            ratio = logratio.exp()

            with torch.no_grad():
                log_ratio = new_log_prob - log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)

            minibatch_advantages = advantages
            minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

            pg_loss = (-minibatch_advantages * ratio).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - self.entropy_coef * entropy_loss

            return ratio, loss, pg_loss, entropy_loss, approx_kl_div
        

        @torch.compile(mode="default")
        def critic_loss_fn(critic, states, returns):
            new_value = critic.get_value(states).reshape(-1)
            v_loss = (0.5 * (new_value - returns) ** 2).mean()

            return self.critic_coef * v_loss, v_loss

        self.set_train_mode()

        batch = Batch(
            states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            next_states = torch.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=torch.float32).to(self.device),
            actions = torch.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=torch.float32).to(self.device),
            rewards = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            values = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            terminations = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            log_probs = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            advantages = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
            returns = torch.zeros((self.nr_steps, self.nr_envs), dtype=torch.float32).to(self.device),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
        

            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                with torch.no_grad():
                    action, processed_action, log_prob = self.policy.get_action_logprob(state)
                    value = self.critic.get_value(state)
                next_state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                actual_next_state = next_state.clone()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = torch.tensor(np.array(self.env.get_final_observation_at_index(info, i), dtype=np.float32), dtype=torch.float32).to(self.device)
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = torch.tensor(reward, dtype=torch.float32).to(self.device)
                batch.values[step] = value.reshape(-1)
                batch.terminations[step]= torch.tensor(terminated, dtype=torch.float32).to(self.device)
                batch.log_probs[step] = log_prob     
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            with torch.no_grad():
                next_values = self.critic.get_value(batch.next_states).squeeze(-1)
            batch.advantages, batch.returns = calculate_gae_advantages_and_returns(batch.rewards, batch.terminations, batch.values, next_values, self.gamma, self.gae_lambda)
            
            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                for param_group in self.policy_optimizer.param_groups:
                    param_group["lr"] = learning_rate
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = learning_rate
            
            batch_states = batch.states.reshape((-1,) + self.os_shape)
            batch_actions = batch.actions.reshape((-1,) + self.as_shape)
            batch_advantages = batch.advantages.reshape(-1)
            batch_returns = batch.returns.reshape(-1)
            batch_values = batch.values.reshape(-1)
            batch_log_probs = batch.log_probs.reshape(-1)

            optimization_metrics_list = []
            for epoch in range(self.max_epochs):
                minibatch_indices = self.rng.choice(self.batch_size, size=self.minibatch_size, replace=False)

                # Policy loss
                ratio, loss1, pg_loss, entropy_loss, approx_kl_div = \
                    policy_loss_fn(self.policy, batch_states[minibatch_indices], batch_actions[minibatch_indices], batch_log_probs[minibatch_indices], batch_advantages[minibatch_indices])
                loss2, critic_loss = critic_loss_fn(self.critic, batch_states[minibatch_indices], batch_returns[minibatch_indices])

                self.policy_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss1.backward()
                loss2.backward()

                policy_grad_norm = 0.0
                critic_grad_norm = 0.0
                for param in self.policy.parameters():
                    policy_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                for param in self.critic.parameters():
                    critic_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                policy_grad_norm = policy_grad_norm ** 0.5
                critic_grad_norm = critic_grad_norm ** 0.5

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                self.critic_optimizer.step()

                ratio_delta = self.delta_calc_operator(torch.abs(ratio - 1))

                # Create metrics
                metrics = {
                    "loss/policy_gradient_loss": pg_loss.item(),
                    "loss/critic_loss": critic_loss.item(),
                    "loss/entropy_loss": entropy_loss.item(),
                    "policy_ratio/ratio_delta": ratio_delta.item(),
                    "policy_ratio/approx_kl": approx_kl_div.item(),
                    "gradients/policy_grad_norm": policy_grad_norm,
                    "gradients/critic_grad_norm": critic_grad_norm,
                }
                optimization_metrics_list.append(metrics)

                if ratio_delta > self.max_ratio_delta:
                    break
            
            y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            optimization_metrics = {key: np.mean([optimization_metrics[key] for optimization_metrics in optimization_metrics_list]) for key in optimization_metrics_list[0].keys()}
            optimization_metrics["optim/nr_epochs"] = epoch + 1
            optimization_metrics["lr/learning_rate"] = learning_rate
            optimization_metrics["v_value/explained_variance"] = explained_var
            optimization_metrics["policy/std_dev"] = np.mean(np.exp(self.policy.policy_logstd.data.cpu().numpy()))

            nr_updates += epoch + 1

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    with torch.no_grad():
                        processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                    state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
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
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time

            time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))


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
            
            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            self.end_logging()


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
        file_path = self.save_path + "/model_best.pt"
        torch.save({
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = ESPO(config, env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.load_state_dict(checkpoint["critic_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        return model


    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                with torch.no_grad():
                    processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
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
