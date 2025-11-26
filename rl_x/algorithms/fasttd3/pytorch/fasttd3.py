import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from rl_x.algorithms.fasttd3.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.fasttd3.pytorch.policy import get_policy
from rl_x.algorithms.fasttd3.pytorch.critic import get_critic
from rl_x.algorithms.fasttd3.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")


class FastTD3:
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
        self.compile_mode = config.algorithm.compile_mode
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.weight_decay = config.algorithm.weight_decay
        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.nr_atoms = config.algorithm.nr_atoms
        self.n_steps = config.algorithm.n_steps
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.noise_std_min = config.algorithm.noise_std_min
        self.noise_std_max = config.algorithm.noise_std_max
        self.smoothing_epsilon = config.algorithm.smoothing_epsilon
        self.smoothing_clip_value = config.algorithm.smoothing_clip_value
        self.nr_critic_updates_per_policy_update = config.algorithm.nr_critic_updates_per_policy_update
        self.nr_policy_updates_per_step = config.algorithm.nr_policy_updates_per_step
        self.clipped_double_q_learning = config.algorithm.clipped_double_q_learning
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.horizon = env.horizon

        if self.evaluation_frequency != -1:
            raise NotImplementedError("Evaluation is not supported yet.")

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

        self.policy = get_policy(config, env, self.device)
        self.critic = get_critic(config, env, self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.q_optimizer = optim.Adam(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.save_model:
            os.makedirs(self.save_path)

    
    def train(self):
        @torch.compile(mode=self.compile_mode)
        def policy_loss_fn(current_log_probs, q1, q2, alpha):
            min_q = torch.minimum(q1, q2)
            policy_loss = (alpha.detach() * current_log_probs - min_q).mean()  # sign switched compared to paper because paper uses gradient ascent

            return policy_loss, min_q
        

        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(critic, states, next_states, actions, next_actions, next_log_probs, rewards, dones, alpha):
            with torch.no_grad():
                next_q1_target = critic.q1_target(next_states, next_actions)
                next_q2_target = critic.q2_target(next_states, next_actions)
                min_next_q_target = torch.minimum(next_q1_target, next_q2_target)
                y = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * (min_next_q_target - alpha.detach() * next_log_probs)

            q1 = critic.q1(states, actions)
            q2 = critic.q2(states, actions)
            q1_loss = F.mse_loss(q1, y)
            q2_loss = F.mse_loss(q2, y)
            q_loss = (q1_loss + q2_loss) / 2

            return q_loss


        self.set_train_mode()

        replay_buffer = ReplayBuffer(
            self.buffer_size_per_env,
            self.nr_envs,
            self.env.single_observation_space.shape,
            self.env.single_action_space.shape,
            self.n_steps,
            self.gamma,
            self.device
        )

        state, _ = self.env.reset()
        noise_scales = torch.rand(self.nr_envs, 1, device=self.device) * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            dones_this_rollout = 0
            action, processed_action = self.policy.get_action(state, noise_scales)
            state, reward, terminated, truncated, info = self.env.step(processed_action)
            done = terminated | truncated
            actual_next_state = state  # TODO: Use the correct actual_next_state
            dones_this_rollout += done.sum().item()
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)

            noise_scales = torch.where(
                done[:, None],
                torch.rand(self.nr_envs, 1, device=self.device) * (self.noise_std_max - self.noise_std_min) + self.noise_std_min,
                noise_scales
            )

            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts * self.nr_envs
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


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
                print("hey")
                exit()
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                # Critic loss
                with torch.no_grad():
                    next_actions, _, next_log_probs = self.policy.get_action(batch_next_states)
                
                q_loss = critic_loss_fn(self.critic, batch_states, batch_next_states, batch_actions, next_actions, next_log_probs, batch_rewards, batch_terminations, self.alpha)

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
                policy_loss, min_q = policy_loss_fn(current_log_probs, q1, q2, self.alpha)

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

                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                # TODO
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                self.save()
            
            saving_end_time = time.time()
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - start_time))


            # Logging
            if should_log:
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
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

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
        file_path = self.save_path + "/latest.model"
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.critic.q1.state_dict(),
            "q2_state_dict": self.critic.q2.state_dict(),
            "q1_target_state_dict": self.critic.q1_target.state_dict(),
            "q2_target_state_dict": self.critic.q2_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
        }
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FastTD3(config, env, run_path, writer)
        model.policy.load_state_dict(checkpoint["policy_state_dict"])
        model.critic.q1.load_state_dict(checkpoint["q1_state_dict"])
        model.critic.q2.load_state_dict(checkpoint["q2_state_dict"])
        model.critic.q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
        model.critic.q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
        model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        model.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        return model

    
    def test(self):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        self.set_eval_mode()
        state, _ = self.env.reset()
        while True:
            with torch.no_grad():
                _, processed_action = self.policy.get_action(state)
            self.env.step(processed_action)


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

    
    def general_properties():
        return GeneralProperties
