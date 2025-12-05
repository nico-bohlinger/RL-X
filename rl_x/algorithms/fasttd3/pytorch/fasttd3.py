import os
import logging
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
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
        self.bf16_mixed_precision_training = config.algorithm.bf16_mixed_precision_training
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
        self.save_frequency = config.algorithm.save_frequency
        self.horizon = env.horizon

        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")
        
        if self.save_frequency != -1 and self.save_frequency % self.nr_envs != 0:
            raise ValueError("The save frequency must be a multiple of the number of environments.")

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

        self.policy = get_policy(config, env, self.device)
        self.critic = get_critic(config, env, self.device)

        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.q_optimizer = optim.AdamW(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.anneal_learning_rate:
            self.q_scheduler = optim.lr_scheduler.LinearLR(self.q_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.total_timesteps // self.nr_envs)
            self.policy_scheduler = optim.lr_scheduler.LinearLR(self.policy_optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.total_timesteps // self.nr_envs)
        
        self.q_support = torch.linspace(self.v_min, self.v_max, self.nr_atoms, device=self.device)

        if self.save_model:
            os.makedirs(self.save_path)

    
    def train(self):
        @torch.compile(mode=self.compile_mode)
        def policy_loss_fn(states):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                actions = self.policy(states)

                q1_dist = F.softmax(self.critic.q1(states, actions), dim=1)
                q2_dist = F.softmax(self.critic.q2(states, actions), dim=1)

                q1_value = torch.sum(q1_dist * self.q_support.unsqueeze(0), dim=1)
                q2_value = torch.sum(q2_dist * self.q_support.unsqueeze(0), dim=1)

                if self.clipped_double_q_learning:
                    q_value = torch.minimum(q1_value, q2_value)
                else:
                    q_value = (q1_value + q2_value) / 2.0

                loss = -q_value.mean()
            
            self.policy_optimizer.zero_grad()
            loss.backward()

            if self.max_grad_norm > 0.0:
                policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            else:
                policy_grad_norm_val = 0.0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        policy_grad_norm_val += p.grad.detach().data.norm(2) ** 2
                policy_grad_norm = policy_grad_norm_val ** 0.5
            
            self.policy_optimizer.step()

            return loss, policy_grad_norm
        

        @torch.compile(mode=self.compile_mode)
        def critic_loss_fn(states, next_states, actions, rewards, dones, truncations, effective_n_steps):
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    noise = torch.clamp(torch.randn_like(actions) * self.smoothing_epsilon, -self.smoothing_clip_value, self.smoothing_clip_value)
                    next_actions = torch.clamp(self.policy(next_states) + noise, -1.0, 1.0)
                    delta_z = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                    bootstrap = 1.0 - (dones * (1.0 - truncations))
                    discount = (self.gamma ** effective_n_steps) * bootstrap
                    target_z = torch.clamp(rewards.unsqueeze(1) + discount.unsqueeze(1) * self.q_support.unsqueeze(0), self.v_min, self.v_max)
                    b = (target_z - self.v_min) / delta_z
                    l = torch.floor(b).long()
                    u = torch.ceil(b).long()

                    is_int = (l == u)
                    l_mask = torch.logical_and(is_int, l > 0)
                    u_mask = torch.logical_and(is_int, l == 0)

                    l = torch.where(l_mask, l - 1, l)
                    u = torch.where(u_mask, u + 1, u)

                    q1_next_dist = F.softmax(self.critic.q1_target(next_states, next_actions), dim=1)
                    q2_next_dist = F.softmax(self.critic.q2_target(next_states, next_actions), dim=1)

                    batch_size = states.shape[0]
                    offset = (
                        torch.linspace(0, (batch_size - 1) * self.nr_atoms, batch_size, device=self.device)
                        .long()
                        .unsqueeze(1)
                        .expand(batch_size, self.nr_atoms)
                    )
                    proj_dist1 = torch.zeros_like(q1_next_dist)
                    proj_dist2 = torch.zeros_like(q2_next_dist)
                    wt_l = (u.float() - b)
                    wt_u = (b - l.float())

                    proj_dist1.view(-1).index_add_(
                        0, (l + offset).view(-1), (q1_next_dist * wt_l).view(-1)
                    )
                    proj_dist1.view(-1).index_add_(
                        0, (u + offset).view(-1), (q1_next_dist * wt_u).view(-1)
                    )
                    proj_dist2.view(-1).index_add_(
                        0, (l + offset).view(-1), (q2_next_dist * wt_l).view(-1)
                    )
                    proj_dist2.view(-1).index_add_(
                        0, (u + offset).view(-1), (q2_next_dist * wt_u).view(-1)
                    )

                    q1_next_value = torch.sum(proj_dist1 * self.q_support.unsqueeze(0), dim=1)
                    q2_next_value = torch.sum(proj_dist2 * self.q_support.unsqueeze(0), dim=1)

                    if self.clipped_double_q_learning:
                        qf_next_target_dist = torch.where(q1_next_value.unsqueeze(1) < q2_next_value.unsqueeze(1), proj_dist1, proj_dist2)
                        qf1_next_target_dist = qf_next_target_dist
                        qf2_next_target_dist = qf_next_target_dist
                    else:
                        qf1_next_target_dist = proj_dist1
                        qf2_next_target_dist = proj_dist2
                        
                current_q1 = self.critic.q1(states, actions)
                current_q2 = self.critic.q2(states, actions)

                q1_loss = -torch.sum(qf1_next_target_dist * F.log_softmax(current_q1, dim=1), dim=1).mean()
                q2_loss = -torch.sum(qf2_next_target_dist * F.log_softmax(current_q2, dim=1), dim=1).mean()
                
                q_loss = q1_loss + q2_loss
                
                # Metrics
                q_min = q1_next_value.min()
                q_max = q1_next_value.max()
                        
            self.q_optimizer.zero_grad()
            q_loss.backward()
            
            if self.max_grad_norm > 0.0:
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()), self.max_grad_norm)
            else:
                critic_grad_norm_val = 0.0
                for p in list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()):
                    if p.grad is not None:
                        critic_grad_norm_val += p.grad.detach().data.norm(2) ** 2
                critic_grad_norm = critic_grad_norm_val ** 0.5
            
            self.q_optimizer.step()
            
            return q_loss, q_min, q_max, critic_grad_norm


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
            torch.compiler.cudagraph_mark_step_begin()


            # Acting
            dones_this_rollout = 0
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                action, processed_action = self.policy.get_action(state, noise_scales)
            next_state, reward, terminated, truncated, info = self.env.step(processed_action)
            done = terminated | truncated
            actual_next_state = next_state  # TODO: Handle this properly
            dones_this_rollout += done.sum().item()
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)

            noise_scales = torch.where(
                done[:, None],
                torch.rand(self.nr_envs, 1, device=self.device) * (self.noise_std_max - self.noise_std_min) + self.noise_std_min,
                noise_scales
            )

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts * self.nr_envs
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0 and global_step % self.save_frequency == 0
            should_log = global_step % self.logging_frequency == 0

            
            # Optimizing
            if should_optimize:
                for _ in range(self.nr_policy_updates_per_step):
                    critic_grad_norm = 0.0
                    policy_grad_norm = 0.0

                    for _ in range(self.nr_critic_updates_per_policy_update):
                        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations, batch_effective_n_steps = replay_buffer.sample(self.batch_size)
                        
                        q_loss, q_min, q_max, critic_grad_norm = critic_loss_fn(batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations, batch_effective_n_steps)
                        
                        with torch.no_grad():
                            for param, target_param in zip(self.critic.q1.parameters(), self.critic.q1_target.parameters()):
                                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)
                            for param, target_param in zip(self.critic.q2.parameters(), self.critic.q2_target.parameters()):
                                target_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

                        nr_updates += 1
                    
                    policy_loss, policy_grad_norm = policy_loss_fn(batch_states)

                    optimization_metrics = {
                        "gradients/policy_grad_norm": policy_grad_norm.item(),
                        "gradients/critic_grad_norm": critic_grad_norm.item(),
                        "loss/q_loss": q_loss.item(),
                        "loss/policy_loss": policy_loss.item(),
                        "lr/learning_rate": self.learning_rate if not self.anneal_learning_rate else self.q_scheduler.get_last_lr()[0],
                        "q/q_max": q_max.item(),
                        "q/q_min": q_min.item()
                    }

                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
            
            if self.anneal_learning_rate:
                self.q_scheduler.step()
                self.policy_scheduler.step()

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.env.reset()
                for _ in range(self.horizon):
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        _, eval_processed_action = self.policy.get_action(eval_state)
                    if self.bf16_mixed_precision_training:
                        eval_processed_action = eval_processed_action.to(torch.float32)
                    eval_state, _, _, _, eval_info = self.env.step(eval_processed_action)
                    eval_logging_info = self.env.get_logging_info_dict(eval_info)
                    if "episode_return" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_return", []).extend(eval_logging_info["episode_return"])
                    if "episode_length" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_length", []).extend(eval_logging_info["episode_length"])
                noise_scales = torch.rand(self.nr_envs, 1, device=self.device) * (self.noise_std_max - self.noise_std_min) + self.noise_std_min
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
                steps_metrics["steps/nr_critic_updates"] = nr_updates
                steps_metrics["steps/nr_policy_updates"] = nr_updates // self.nr_critic_updates_per_policy_update
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
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
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
