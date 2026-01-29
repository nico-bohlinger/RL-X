import os
import logging
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
import wandb

from rl_x.algorithms.mpo.pytorch.general_properties import GeneralProperties
from rl_x.algorithms.mpo.pytorch.policy import get_policy
from rl_x.algorithms.mpo.pytorch.critic import get_critic
from rl_x.algorithms.mpo.pytorch.dual_variables import DualVariables
from rl_x.algorithms.mpo.pytorch.replay_buffer import ReplayBuffer
from rl_x.algorithms.mpo.pytorch.observation_normalizer import get_observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class MPO:
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
        self.agent_learning_rate = config.algorithm.agent_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_agent_learning_rate = config.algorithm.anneal_agent_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.actor_update_period = config.algorithm.actor_update_period
        self.target_network_update_period = config.algorithm.target_network_update_period
        self.gamma = config.algorithm.gamma
        self.n_steps = config.algorithm.n_steps
        self.optimize_every_n_steps = config.algorithm.optimize_every_n_steps
        self.action_sampling_number = config.algorithm.action_sampling_number
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.epsilon_non_parametric = config.algorithm.epsilon_non_parametric
        self.epsilon_parametric_mu = config.algorithm.epsilon_parametric_mu
        self.epsilon_parametric_sigma = config.algorithm.epsilon_parametric_sigma
        self.epsilon_penalty = config.algorithm.epsilon_penalty
        self.action_clipping = config.algorithm.action_clipping
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.nr_atoms = config.algorithm.nr_atoms
        self.float_epsilon = config.algorithm.float_epsilon
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

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

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        self.actor = get_policy(config, self.train_env, self.device)
        self.target_actor = get_policy(config, self.train_env, self.device)
        self.env_actor = get_policy(config, self.train_env, self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.env_actor.load_state_dict(self.target_actor.state_dict())

        self.critic = get_critic(config, self.train_env, self.device)

        nr_actions = int(np.prod(self.train_env.single_action_space.shape, dtype=int).item())
        self.duals = DualVariables(config, nr_actions, self.device)

        fused = self.device.type == "cuda"
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.agent_learning_rate, fused=fused)
        self.critic_optimizer = optim.Adam(self.critic.q.parameters(), lr=self.agent_learning_rate, fused=fused)
        self.dual_optimizer = optim.Adam(self.duals.parameters(), lr=self.dual_learning_rate, fused=fused)

        if self.anneal_agent_learning_rate:
            self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)
            self.actor_scheduler = optim.lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)
        if self.anneal_dual_learning_rate:
            self.dual_scheduler = optim.lr_scheduler.LinearLR(self.dual_optimizer, start_factor=1.0, end_factor=0.0, total_iters=(self.total_timesteps - self.learning_starts) // self.nr_envs)

        self.observation_normalizer = get_observation_normalizer(config, self.train_env.single_observation_space.shape[0], self.device)

        self.q_support = torch.linspace(self.v_min, self.v_max, self.nr_atoms, device=self.device)
        self.log_num_actions = torch.log(torch.tensor(self.action_sampling_number, dtype=torch.float32, device=self.device))
        self.log_2pi = torch.log(torch.tensor(2.0 * np.pi, dtype=torch.float32, device=self.device))

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf

    
    def train(self):
        @torch.compile(mode=self.compile_mode)
        def update(states, next_states, actions, rewards, dones, truncations, effective_n_steps):
            # Critic update
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                with torch.no_grad():
                    target_next_action_mean, target_next_action_std = self.target_actor.get_action(next_states)
                    sampled_next_actions = target_next_action_mean.unsqueeze(0) + target_next_action_std.unsqueeze(0) * torch.randn((self.action_sampling_number, target_next_action_mean.shape[0], target_next_action_mean.shape[1]), device=self.device)  # (sampled actions, batch, action_dim)

                    expanded_next_states = next_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)  # (sampled actions, batch, state_dim)
                    target_next_logits = self.critic.q_target(expanded_next_states.reshape(-1, expanded_next_states.shape[-1]), sampled_next_actions.reshape(-1, sampled_next_actions.shape[-1]))
                    target_next_logits = target_next_logits.view(self.action_sampling_number, states.shape[0], self.nr_atoms)  # (sampled actions, batch, atoms)
                    target_next_pmf = F.softmax(target_next_logits, dim=-1).unsqueeze(-2)  # (sampled actions, batch, 1, atoms)

                    bootstrap = 1.0 - (dones * (1.0 - truncations))
                    discount = (self.gamma ** effective_n_steps) * bootstrap
                    target_z = torch.clamp(rewards.unsqueeze(1) + discount.unsqueeze(1) * self.q_support.unsqueeze(0), self.v_min, self.v_max)  # (batch, atoms)

                    abs_delta = torch.abs(
                        target_z.unsqueeze(1) - self.q_support.unsqueeze(0).unsqueeze(2)  # (1, atoms, 1)
                    ).unsqueeze(0)  # (1, batch, atoms, atoms)

                    delta_z = self.q_support[1] - self.q_support[0]
                    target_pmf = torch.clamp(1.0 - abs_delta / delta_z, 0.0, 1.0) * target_next_pmf  # (sampled actions, batch, atoms, atoms)
                    target_pmf = torch.sum(target_pmf, dim=-1)  # (sampled actions, batch, atoms)
                    target_pmf = target_pmf.mean(0)  # (batch, atoms)

                current_logits = self.critic.q(states, actions)  # (batch, atoms)
                current_q = (F.softmax(current_logits, dim=-1) * self.q_support).sum(-1)  # (batch,)
                q_loss = -torch.sum(target_pmf * F.log_softmax(current_logits, dim=1), dim=1).mean()

            self.critic_optimizer.zero_grad()
            q_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.q.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            # Actor and dual update
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                stacked_states = torch.cat([states, next_states], dim=0)  # (2 * batch, state_dim)

                with torch.no_grad():
                    target_action_mean, target_action_std = self.target_actor.get_action(stacked_states)
                    sampled_actions = target_action_mean.unsqueeze(0) + target_action_std.unsqueeze(0) * torch.randn((self.action_sampling_number, target_action_mean.shape[0], target_action_mean.shape[1]), device=self.device)  # (sampled actions, 2 * batch, action_dim)

                    expanded_states = stacked_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)  # (sampled actions, 2 * batch, state_dim)
                    expanded_stacked_logits = self.critic.q_target(
                        expanded_states.reshape(-1, expanded_states.shape[-1]),
                        sampled_actions.reshape(-1, sampled_actions.shape[-1]),
                    ).view(self.action_sampling_number, stacked_states.shape[0], self.nr_atoms)  # (sampled actions, 2 * batch, atoms)

                    expanded_stacked_pmf = torch.softmax(expanded_stacked_logits, dim=-1)
                    expanded_stacked_q = (expanded_stacked_pmf * self.q_support).sum(-1)  # (sampled actions, 2 * batch)
                
                eta = F.softplus(self.duals.log_eta) + self.float_epsilon
                improvement_dist = F.softmax(expanded_stacked_q / eta.detach(), dim=0)  # (sampled actions, 2 * batch)

                q_logsumexp = torch.logsumexp(expanded_stacked_q / eta, dim=0)  # (2 * batch,)
                loss_eta = eta * (self.epsilon_non_parametric + torch.mean(q_logsumexp) - self.log_num_actions)

                if self.action_clipping:
                    penalty_temperature = F.softplus(self.duals.log_penalty_temperature) + self.float_epsilon
                    diff_oob = sampled_actions - torch.clamp(sampled_actions, -1.0, 1.0)  # (sampled actions, 2 * batch, action_dim)
                    cost_oob = -torch.linalg.norm(diff_oob, dim=-1)  # (sampled actions, 2 * batch)
                    penalty_improvement = F.softmax(cost_oob / penalty_temperature.detach(), dim=0)

                    penalty_logsumexp = torch.logsumexp(cost_oob / penalty_temperature, dim=0)  # (2 * batch,)
                    loss_penalty_temp = penalty_temperature * (self.epsilon_penalty + torch.mean(penalty_logsumexp) - self.log_num_actions)

                    improvement_dist = improvement_dist + penalty_improvement
                    loss_eta = loss_eta + loss_penalty_temp

                    penalty_temperature_detached = penalty_temperature.detach()
                else:
                    penalty_temperature_detached = torch.tensor(0.0, device=self.device)

                online_action_mean, online_action_std = self.actor.get_action(stacked_states)

                # This is the same as:
                # F.softplus(self.duals.log_alpha_mean) + self.float_epsilon
                # But softplus in Torch has a threshold value which switches to a linear approximation for large values
                alpha_mean = (torch.logaddexp(self.duals.log_alpha_stddev, torch.zeros_like(self.duals.log_alpha_stddev)) + self.float_epsilon)
                alpha_std = (torch.logaddexp(self.duals.log_alpha_stddev, torch.zeros_like(self.duals.log_alpha_stddev)) + self.float_epsilon)

                logprob_mean = torch.sum(-0.5 * ((((sampled_actions - online_action_mean) / target_action_std) ** 2) + self.log_2pi) - torch.log(target_action_std), dim=-1)  # (sampled actions, 2 * batch)

                loss_pg_mean = -(logprob_mean * improvement_dist).sum(dim=0).mean()

                kl_mean_std0 = torch.clamp(target_action_std, min=self.float_epsilon)
                kl_mean_std1 = torch.clamp(target_action_std, min=self.float_epsilon)
                kl_mean_var0 = kl_mean_std0 ** 2
                kl_mean_var1 = kl_mean_std1 ** 2
                kl_mean = torch.log(kl_mean_std1 / kl_mean_std0) + (kl_mean_var0 + (target_action_mean - online_action_mean) ** 2) / (2.0 * kl_mean_var1) - 0.5
                mean_kl_mean = kl_mean.mean(dim=0)  # (action_dim,)
                loss_kl_mean = torch.sum(alpha_mean.detach() * mean_kl_mean)
                loss_alpha_mean = torch.sum(alpha_mean * (self.epsilon_parametric_mu - mean_kl_mean.detach()))

                logprob_std = torch.sum(-0.5 * ((((sampled_actions - target_action_mean) / online_action_std) ** 2) + self.log_2pi) - torch.log(online_action_std), dim=-1)  # (sampled actions, 2 * batch)

                loss_pg_std = -(logprob_std * improvement_dist).sum(dim=0).mean()

                kl_std_std0 = torch.clamp(target_action_std, min=self.float_epsilon)
                kl_std_std1 = torch.clamp(online_action_std, min=self.float_epsilon)
                kl_std_var0 = kl_std_std0 ** 2
                kl_std_var1 = kl_std_std1 ** 2
                kl_std = torch.log(kl_std_std1 / kl_std_std0) + (kl_std_var0 + (target_action_mean - target_action_mean) ** 2) / (2.0 * kl_std_var1) - 0.5
                mean_kl_std = kl_std.mean(dim=0)
                loss_kl_std = torch.sum(alpha_std.detach() * mean_kl_std)
                loss_alpha_std = torch.sum(alpha_std * (self.epsilon_parametric_sigma - mean_kl_std.detach()))

                actor_loss = loss_pg_mean + loss_pg_std + loss_kl_mean + loss_kl_std
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta
            self.dual_optimizer.zero_grad()
            dual_loss.backward()
            dual_grad_norm = torch.nn.utils.clip_grad_norm_(self.duals.parameters(), self.max_grad_norm)
            self.dual_optimizer.step()
        
            with torch.no_grad():
                self.duals.log_eta.data.clamp_(min=self.min_log_temperature)
                self.duals.log_alpha_mean.data.clamp_(min=self.min_log_alpha)
                self.duals.log_alpha_stddev.data.clamp_(min=self.min_log_alpha)

            return (
                q_loss,
                actor_loss,
                dual_loss,
                current_q.mean(),
                eta.detach(),
                penalty_temperature_detached,
                alpha_mean.detach().mean(),
                alpha_std.detach().mean(),
                loss_eta.detach(),
                loss_alpha_mean.detach() + loss_alpha_std.detach(),
                mean_kl_mean.detach().mean(),
                mean_kl_std.detach().mean(),
                actor_grad_norm.detach(),
                critic_grad_norm.detach(),
                dual_grad_norm.detach(),
                online_action_std.detach().min(dim=1).values.mean(),
                online_action_std.detach().max(dim=1).values.mean(),
            )


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(max(1, self.buffer_size // self.nr_envs)), self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma,  self.device)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
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
                action = torch.tensor((processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0, dtype=torch.float32, device=self.device)
            else:
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    normalized_state = self.observation_normalizer.normalize(state, update=False)
                    action, processed_action = self.env_actor.sample_action(normalized_state)
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
            
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            actual_next_state = torch.tensor(actual_next_state, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            # terminated = torch.tensor(terminated, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)
            truncated = torch.tensor(truncated, dtype=torch.float32, device=self.device)
            replay_buffer.add(state, actual_next_state, action, reward, done, truncated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            iteration = global_step // self.nr_envs
            should_learning_start = global_step > self.learning_starts
            should_update_actor = should_learning_start and (iteration % self.actor_update_period == 0)
            should_optimize = should_learning_start and (iteration % self.optimize_every_n_steps == 0)
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0

            
            # Update actor
            if should_update_actor:
                self.env_actor.load_state_dict(self.target_actor.state_dict())


            # Optimizing
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncation, batch_effective_n_steps = replay_buffer.sample(self.batch_size)
                batch_normalized_states = self.observation_normalizer.normalize(batch_states, update=True)
                batch_normalized_next_states = self.observation_normalizer.normalize(batch_next_states, update=True)

                (
                    q_loss,
                    actor_loss,
                    dual_loss,
                    current_q_mean,
                    eta,
                    penalty_temperature,
                    alpha_mean,
                    alpha_std,
                    dual_loss_eta,
                    dual_loss_alpha,
                    mean_kl_mean,
                    mean_kl_std,
                    actor_grad_norm,
                    critic_grad_norm,
                    dual_grad_norm,
                    min_action_stddev,
                    max_action_stddev,
                ) = update(
                    batch_normalized_states,
                    batch_normalized_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_dones,
                    batch_truncation,
                    batch_effective_n_steps,
                )
                nr_updates += 1

                if nr_updates % self.target_network_update_period == 0:
                    self.target_actor.load_state_dict(self.actor.state_dict())
                    self.critic.q_target.load_state_dict(self.critic.q.state_dict())

                # Create metrics
                optimization_metrics = {
                    "loss/critic_loss": q_loss.item(),
                    "loss/actor_loss": actor_loss.item(),
                    "loss/dual_loss": dual_loss.item(),
                    "loss/loss_eta": dual_loss_eta.item(),
                    "loss/loss_alpha": dual_loss_alpha.item(),
                    "q/current_q_mean": current_q_mean.item(),
                    "dual/eta": eta.item(),
                    "dual/penalty_temperature": penalty_temperature.item(),
                    "dual/alpha_mean": alpha_mean.item(),
                    "dual/alpha_std": alpha_std.item(),
                    "kl/mean_kl_mean": mean_kl_mean.item(),
                    "kl/mean_kl_std": mean_kl_std.item(),
                    "gradients/actor_grad_norm": actor_grad_norm.item(),
                    "gradients/critic_grad_norm": critic_grad_norm.item(),
                    "gradients/dual_grad_norm": dual_grad_norm.item(),
                    "policy/std_min_mean": min_action_stddev.item(),
                    "policy/std_max_mean": max_action_stddev.item(),
                }

                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
            
                if self.anneal_agent_learning_rate:
                    self.critic_scheduler.step()
                    self.actor_scheduler.step()
                if self.anneal_dual_learning_rate:
                    self.dual_scheduler.step()
            
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
                        eval_normalized_state = self.observation_normalizer.normalize(torch.tensor(eval_state, dtype=torch.float32, device=self.device), update=False)
                        eval_processed_action = self.actor.get_deterministic_action(eval_normalized_state)
                    if self.bf16_mixed_precision_training:
                        eval_processed_action = eval_processed_action.to(torch.float32)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(eval_processed_action.cpu().numpy())
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
            
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


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
        file_path = self.save_path + "/best.model"
        save_dict = {
            "config_algorithm": self.config.algorithm,
            "actor_state_dict": self.actor.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "critic_state_dict": self.critic.q.state_dict(),
            "critic_target_state_dict": self.critic.q_target.state_dict(),
            "duals_state_dict": self.duals.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "dual_optimizer_state_dict": self.dual_optimizer.state_dict(),
            "observation_normalizer_state_dict": self.observation_normalizer.state_dict(),
        }
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))
    

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        checkpoint = torch.load(config.runner.load_model, weights_only=False)
        loaded_algorithm_config = checkpoint["config_algorithm"]
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = MPO(config, train_env, eval_env, run_path, writer)
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        model.env_actor.load_state_dict(checkpoint["target_actor_state_dict"])
        model.critic.q.load_state_dict(checkpoint["critic_state_dict"])
        model.critic.q_target.load_state_dict(checkpoint["critic_target_state_dict"])
        model.duals.load_state_dict(checkpoint["duals_state_dict"])
        model.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        model.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        model.dual_optimizer.load_state_dict(checkpoint["dual_optimizer_state_dict"])
        model.observation_normalizer.load_state_dict(checkpoint["observation_normalizer_state_dict"])
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
                    normalized_state = self.observation_normalizer.normalize(torch.tensor(state, dtype=torch.float32, device=self.device), update=False)
                    processed_action = self.actor.get_deterministic_action(normalized_state)
                state, reward, terminated, truncated, info = self.eval_env.step(processed_action.cpu().numpy())
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        self.actor.train()
        self.target_actor.train()
        self.env_actor.train()
        self.critic.q.train()
        self.critic.q_target.train()
        self.duals.train()


    def set_eval_mode(self):
        self.actor.eval()
        self.target_actor.eval()
        self.env_actor.eval()
        self.critic.q.eval()
        self.critic.q_target.eval()
        self.duals.eval()

    
    def general_properties():
        return GeneralProperties
