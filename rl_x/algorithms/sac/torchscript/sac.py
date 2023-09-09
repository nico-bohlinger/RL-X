import os
import logging
import random
import time
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import wandb

from rl_x.algorithms.sac.torchscript.policy import get_policy
from rl_x.algorithms.sac.torchscript.critic import get_critic
from rl_x.algorithms.sac.torchscript.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.sac.torchscript.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class SAC():
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
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.target_entropy = config.algorithm.target_entropy
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        self.device = torch.device("cuda" if config.algorithm.device == "gpu" and torch.cuda.is_available() else "cpu")
        rlx_logger.info(f"Using device: {self.device}")

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

        self.env_as_low = torch.tensor(env.action_space.low, dtype=torch.float32).to(self.device)
        self.env_as_high = torch.tensor(env.action_space.high, dtype=torch.float32).to(self.device)

        self.policy = torch.jit.script(get_policy(config, env, self.device).to(self.device))
        self.critic = torch.jit.script(get_critic(config, env, self.device).to(self.device))

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.q_optimizer = optim.Adam(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate)

        self.entropy_coefficient = torch.jit.script(EntropyCoefficient(config, env, self.device).to(self.device))
        self.alpha = self.entropy_coefficient()
        self.entropy_optimizer = optim.Adam([self.entropy_coefficient.log_alpha], lr=self.learning_rate)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape, self.device)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        time_metrics_buffer = deque(maxlen=self.logging_freq)
        optimization_metrics_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}


            # Acting
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low.cpu().numpy()) / (self.env_as_high.cpu().numpy() - self.env_as_low.cpu().numpy()) * 2.0 - 1.0
            else:
                action, processed_action, _ = self.policy.get_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                action = action.detach().cpu().numpy()
                processed_action = processed_action.detach().cpu().numpy()
            
            next_state, reward, terminated, truncated, info = self.env.step(processed_action)
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    maybe_final_observation = self.env.get_final_observation(info, i)
                    if maybe_final_observation is not None:
                        actual_next_state[i] = maybe_final_observation
                    nr_episodes += 1
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs

            episode_infos = self.env.get_episode_infos(info)
            episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_infos if "r" in ep_info])

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_try_to_save = should_learning_start and self.save_model and episode_infos
            should_log = global_step % self.logging_freq == 0


            # Optimizing - Anneal learning rate
            learning_rate = self.learning_rate
            if self.anneal_learning_rate:
                fraction = 1 - (global_step / self.total_timesteps)
                learning_rate = fraction * self.learning_rate
                param_groups = self.policy_optimizer.param_groups + self.q_optimizer.param_groups + self.entropy_optimizer.param_groups
                for param_group in param_groups:
                    param_group["lr"] = learning_rate

            
            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                # Critic loss
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(batch_next_states)
                
                q_loss = self.critic.loss(batch_states, batch_next_states, batch_actions, next_actions, next_log_probs, batch_rewards, batch_terminations, self.alpha)

                self.q_optimizer.zero_grad()
                q_loss.backward()
                q1_grad_norm = 0.0
                q2_grad_norm = 0.0
                for param in self.critic.q1.parameters():
                    q1_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                for param in self.critic.q2.parameters():
                    q2_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                critic_grad_norm = q1_grad_norm ** 0.5 + q2_grad_norm ** 0.5
                self.q_optimizer.step()

                # Update critic targets
                for param, target_param in zip(self.critic.q1.parameters(), self.critic.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic.q2.parameters(), self.critic.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # Policy loss
                current_actions, _, current_log_probs = self.policy.get_action(batch_states)
                q1 = self.critic.q1(batch_states, current_actions)
                q2 = self.critic.q2(batch_states, current_actions)
                policy_loss, min_q = self.policy.loss(current_log_probs, q1, q2, self.alpha)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_grad_norm = 0.0
                for param in self.policy.parameters():
                    policy_grad_norm += param.grad.detach().data.norm(2).item() ** 2
                policy_grad_norm = policy_grad_norm ** 0.5
                self.policy_optimizer.step()

                # Entropy loss
                entropy = -current_log_probs.detach().mean()
                entropy_loss = self.entropy_coefficient.loss(entropy)

                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                entropy_grad_norm = self.entropy_coefficient.log_alpha.grad.detach().data.norm(2).item() ** 2
                self.entropy_optimizer.step()

                self.alpha = self.entropy_coefficient()

                # Create metrics
                optimization_metrics = {
                    "entropy/alpha": self.alpha.item(),
                    "entropy/entropy": entropy.item(),
                    "gradients/policy_grad_norm": policy_grad_norm,
                    "gradients/critic_grad_norm": critic_grad_norm,
                    "gradients/entropy_grad_norm": entropy_grad_norm,
                    "loss/q_loss": q_loss.item(),
                    "loss/policy_loss": policy_loss.item(),
                    "loss/entropy_loss": entropy_loss.item(),
                    "lr/learning_rate": learning_rate,
                    "q_value/q_value": min_q.mean().item(),
                }

                optimization_metrics_buffer.append(optimization_metrics)
                nr_updates += 1
            
            optimize_end_time = time.time()
            time_metrics["time/optimize_time"] = optimize_end_time - acting_end_time


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - optimize_end_time

            time_metrics["time/fps"] = self.nr_envs / (saving_end_time - start_time)

            time_metrics_buffer.append(time_metrics)


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes

                if len(episode_info_buffer) > 0:
                    self.log("rollout/episode_reward", np.mean([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info]), global_step)
                    self.log("rollout/episode_length", np.mean([ep_info["l"] for ep_info in episode_info_buffer if "r" in ep_info]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info]), global_step)
                mean_time_metrics = {key: np.mean([metrics[key] for metrics in time_metrics_buffer]) for key in time_metrics_buffer[0].keys()}
                mean_optimization_metrics = {} if not should_learning_start else {key: np.mean([metrics[key] for metrics in optimization_metrics_buffer]) for key in optimization_metrics_buffer[0].keys()}
                combined_metrics = {**steps_metrics, **mean_time_metrics, **mean_optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                episode_info_buffer.clear()
                time_metrics_buffer.clear()
                optimization_metrics_buffer.clear()

                self.end_logging()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │")

    
    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        file_path = self.save_path + "/model_best.pt"
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.critic.q1.state_dict(),
            "q2_state_dict": self.critic.q2.state_dict(),
            "q1_target_state_dict": self.critic.q1_target.state_dict(),
            "q2_target_state_dict": self.critic.q2_target.state_dict(),
            "log_alpha": self.entropy_coefficient.log_alpha,
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "entropy_optimizer_state_dict": self.entropy_optimizer.state_dict(),
        }
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, run_path, writer):
        checkpoint = torch.load(config.runner.load_model)
        config.algorithm = checkpoint["config_algorithm"]
        model = SAC(config, env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.q1.load_state_dict(checkpoint["q1_state_dict"])
        model.critic.q2.load_state_dict(checkpoint["q2_state_dict"])
        model.critic.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        model.critic.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        model.entropy_coefficient.log_alpha = checkpoint["log_alpha"]
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        model.entropy_optimizer.load_state_dict(checkpoint["entropy_optimizer_state_dict"])
        return model

    
    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = self.policy.get_deterministic_action(torch.tensor(state, dtype=torch.float32).to(self.device))
                state, reward, terminated, truncated, info = self.env.step(processed_action.cpu().numpy())
                done = terminated | truncated
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")


    def set_train_mode(self):
        self.policy.train()
        self.critic.q1.train()
        self.critic.q2.train()
        self.critic.q1_target.train()
        self.critic.q2_target.train()


    def set_eval_mode(self):
        self.policy.eval()
        self.critic.q1.eval()
        self.critic.q2.eval()
        self.critic.q1_target.eval()
        self.critic.q2_target.eval()
