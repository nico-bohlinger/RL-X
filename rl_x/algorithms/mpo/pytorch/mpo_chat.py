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
from rl_x.algorithms.mpo.pytorch.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")

_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0


class DualVariables(torch.nn.Module):
    def __init__(self, act_dim: int, device: torch.device):
        super().__init__()
        # Initialize similar to CleanRL
        self.log_eta = torch.nn.Parameter(torch.tensor([10.0], device=device))
        self.log_alpha_mean = torch.nn.Parameter(torch.tensor([10.0] * act_dim, device=device))
        self.log_alpha_stddev = torch.nn.Parameter(torch.tensor([1000.0] * act_dim, device=device))
        self.log_penalty_temperature = torch.nn.Parameter(torch.tensor([10.0], device=device))


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

        # algo cfg
        self.total_timesteps = int(config.algorithm.total_timesteps)
        self.nr_envs = int(config.environment.nr_envs)
        self.buffer_size = int(config.algorithm.buffer_size)
        self.learning_starts = int(config.algorithm.learning_starts)
        self.batch_size = int(config.algorithm.batch_size)

        self.policy_q_lr = float(config.algorithm.policy_q_lr)
        self.dual_lr = float(config.algorithm.dual_lr)
        self.anneal_learning_rate = bool(config.algorithm.anneal_learning_rate)

        self.gamma = float(config.algorithm.gamma)
        self.n_steps = int(config.algorithm.n_steps)

        self.optimize_every = int(config.algorithm.optimize_every)
        self.target_network_update_period = int(config.algorithm.target_network_update_period)  # SGD steps
        self.variable_update_period = int(config.algorithm.variable_update_period)              # env steps

        self.action_sampling_number = int(config.algorithm.action_sampling_number)
        self.grad_norm_clip = float(config.algorithm.grad_norm_clip)

        self.eps_nonparam = float(config.algorithm.epsilon_non_parametric)
        self.eps_mu = float(config.algorithm.epsilon_parametric_mu)
        self.eps_sigma = float(config.algorithm.epsilon_parametric_sigma)
        self.eps_penalty = float(config.algorithm.epsilon_penalty)

        self.vmax = float(config.algorithm.vmax_values)
        self.num_bins = int(config.algorithm.categorical_num_bins)

        self.logging_frequency = int(config.algorithm.logging_frequency)
        self.evaluation_frequency = int(config.algorithm.evaluation_frequency)
        self.evaluation_episodes = int(config.algorithm.evaluation_episodes)

        # device
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

        # seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True

        # env action bounds
        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        # models
        self.actor = get_policy(config, self.train_env, self.device)
        self.target_actor = get_policy(config, self.train_env, self.device)
        self.env_actor = get_policy(config, self.train_env, self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.env_actor.load_state_dict(self.target_actor.state_dict())

        self.critic = get_critic(config, self.train_env, self.device)

        act_dim = int(np.prod(self.train_env.single_action_space.shape, dtype=int).item())
        self.duals = DualVariables(act_dim, self.device)

        # optimizers
        fused_ok = (self.device.type == "cuda")
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.policy_q_lr, fused=fused_ok)
        self.critic_optimizer = optim.Adam(self.critic.q.parameters(), lr=self.policy_q_lr, fused=fused_ok)
        self.dual_optimizer = optim.Adam(self.duals.parameters(), lr=self.dual_lr, fused=fused_ok)

        if self.anneal_learning_rate:
            total_iters = max(1, self.total_timesteps // self.nr_envs)
            self.actor_scheduler = optim.lr_scheduler.LinearLR(self.actor_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)
            self.critic_scheduler = optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)
            self.dual_scheduler = optim.lr_scheduler.LinearLR(self.dual_optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_iters)

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.best_mean_return = -np.inf

    def train(self):
        # Distributional support
        v_min = -self.vmax
        v_max = self.vmax
        rd_support = torch.linspace(v_min, v_max, steps=self.num_bins, device=self.device)  # (D,)
        delta_z = rd_support[1] - rd_support[0]
        log_num_actions = torch.log(torch.tensor(float(self.action_sampling_number), device=self.device))

        # Replay buffer (per-env capacity)
        buffer_size_per_env = max(1, self.buffer_size // self.nr_envs)
        replay_buffer = ReplayBuffer(
            buffer_size_per_env=buffer_size_per_env,
            nr_envs=self.nr_envs,
            os_shape=self.train_env.single_observation_space.shape,
            as_shape=self.train_env.single_action_space.shape,
            n_steps=self.n_steps,
            gamma=self.gamma,
            device=self.device,
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        # Scheduling with counters (avoids “multiple of nr_envs” pitfalls)
        next_log_step = self.logging_frequency
        next_eval_step = self.evaluation_frequency if self.evaluation_frequency != -1 else None
        next_env_actor_update_step = self.variable_update_period

        sgd_steps = 0
        nr_updates = 0
        nr_episodes = 0

        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        state, _ = self.train_env.reset()
        global_step = 0  # counts *env steps total* (nr_envs added each rollout)

        @torch.compile(mode=self.compile_mode)
        def train_step(
            states, next_states, actions, rewards, dones, truncations, effective_n_steps
        ):
            """
            MPO update:
              1) critic distributional cross-entropy (TD(n) w/ bootstrap on truncation)
              2) nonparametric improvement distribution (eta + penalty temperature)
              3) parametric regression (mean+std) with KL constraints (alpha_mean/std)
            """
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                # ----------------------------
                # PHASE 1: Critic update (distributional TD(n))
                # ----------------------------
                with torch.no_grad():
                    # sample N actions from target_actor at next_states
                    target_mean, target_std = self.target_actor(next_states)
                    target_dist = torch.distributions.Independent(torch.distributions.Normal(target_mean, target_std), 1)
                    taus = target_dist.rsample((self.action_sampling_number,))  # (N,B,A)
                    taus = torch.clamp(taus, -1.0, 1.0)

                    completed_next_states = next_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)  # (N,B,obs)
                    target_logits = self.critic.q_target(completed_next_states.reshape(-1, completed_next_states.shape[-1]),
                                                         taus.reshape(-1, taus.shape[-1]))
                    target_logits = target_logits.view(self.action_sampling_number, states.shape[0], self.num_bins)  # (N,B,D)

                    next_pmfs = F.softmax(target_logits, dim=-1).unsqueeze(-2)  # (N,B,1,D)

                    # bootstrap on truncation, NOT on termination:
                    # dones = termination flag (1 means terminal -> no bootstrap)
                    tz_hat = rewards.unsqueeze(-1) + (self.gamma ** effective_n_steps).unsqueeze(-1) * rd_support * (1.0 - dones).unsqueeze(-1)  # (B,D)

                    tz_hat_clipped = tz_hat.clamp(v_min, v_max).unsqueeze(-2)  # (B,1,D)

                    # abs delta between projected atoms and support atoms
                    # yields (B,D,D) then broadcast to (N,B,D,D)
                    abs_delta = torch.abs(
                        tz_hat_clipped - rd_support.unsqueeze(0).unsqueeze(-1)  # (1,D,1)
                    ).unsqueeze(0)  # (1,B,D,D)

                    target_pmfs = torch.clamp(1.0 - abs_delta / delta_z, 0.0, 1.0) * next_pmfs  # (N,B,D,D)
                    target_pmfs = torch.sum(target_pmfs, dim=-1)  # (N,B,D)
                    target_pmfs = target_pmfs.mean(0)             # (B,D)

                old_logits = self.critic.q(states, actions)        # (B,D)
                old_logpmf = F.log_softmax(old_logits, dim=-1)     # (B,D)
                old_pmf = F.softmax(old_logits, dim=-1)            # (B,D)
                old_q = (old_pmf * rd_support).sum(-1)             # (B,)

                q_loss = (-(target_pmfs * old_logpmf).sum(-1)).mean()

            # critic step
            self.critic_optimizer.zero_grad(set_to_none=True)
            q_loss.backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.q.parameters(), self.grad_norm_clip)
            self.critic_optimizer.step()

            # ----------------------------
            # PHASE 2+3: MPO actor + duals
            # ----------------------------
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                stacked_states = torch.cat([states, next_states], dim=0)  # (2B,obs)

                # Sample from target policy for improvement distribution
                with torch.no_grad():
                    t_mean, t_std = self.target_actor(stacked_states)
                    target_dist_full = torch.distributions.Independent(torch.distributions.Normal(t_mean, t_std), 1)
                    target_base_per_dim = torch.distributions.Normal(t_mean, t_std)  # (2B,A) base dist for KL

                    sampled_actions = target_dist_full.rsample((self.action_sampling_number,))  # (N,2B,A)
                    sampled_actions = torch.clamp(sampled_actions, -1.0, 1.0)

                    completed_states = stacked_states.unsqueeze(0).expand(self.action_sampling_number, -1, -1)  # (N,2B,obs)
                    logits_sa = self.critic.q_target(
                        completed_states.reshape(-1, completed_states.shape[-1]),
                        sampled_actions.reshape(-1, sampled_actions.shape[-1]),
                    ).view(self.action_sampling_number, stacked_states.shape[0], self.num_bins)  # (N,2B,D)

                    pmf_sa = torch.softmax(logits_sa, dim=-1)
                    q_sa = (pmf_sa * rd_support).sum(-1)  # (N,2B)

                # eta (temperature) + nonparametric distribution
                eta = F.softplus(self.duals.log_eta) + _MPO_FLOAT_EPSILON
                impr_distr = F.softmax(q_sa / eta.detach(), dim=0)  # (N,2B)

                q_logsumexp = torch.logsumexp(q_sa / eta, dim=0)  # (2B,)
                loss_eta = eta * (self.eps_nonparam + torch.mean(q_logsumexp) - log_num_actions)

                # action range penalty (MO-MPO style, as in CleanRL)
                penalty_temperature = F.softplus(self.duals.log_penalty_temperature) + _MPO_FLOAT_EPSILON
                diff_oob = sampled_actions - torch.clamp(sampled_actions, -1.0, 1.0)  # (N,2B,A) (usually 0 after clamp)
                cost_oob = -torch.linalg.norm(diff_oob, dim=-1)  # (N,2B)
                penalty_impr = F.softmax(cost_oob / penalty_temperature.detach(), dim=0)

                penalty_logsumexp = torch.logsumexp(cost_oob / penalty_temperature, dim=0)  # (2B,)
                loss_penalty_temp = penalty_temperature * (self.eps_penalty + torch.mean(penalty_logsumexp) - log_num_actions)

                impr_distr = impr_distr + penalty_impr
                loss_eta = loss_eta + loss_penalty_temp

                # Online actor outputs
                o_mean, o_std = self.actor(stacked_states)

                # dual alphas
                alpha_mean = F.softplus(self.duals.log_alpha_mean) + _MPO_FLOAT_EPSILON
                alpha_std = (torch.logaddexp(self.duals.log_alpha_stddev, torch.zeros_like(self.duals.log_alpha_stddev))
                             + _MPO_FLOAT_EPSILON)

                # --- mean regression (fix std to target std) ---
                online_mean_dist = torch.distributions.Independent(torch.distributions.Normal(o_mean, t_std), 1)
                logp_mean = online_mean_dist.log_prob(sampled_actions)  # (N,2B)

                loss_pg_mean = -(logp_mean * impr_distr).sum(dim=0).mean()

                # KL per dimension for mean (Normal(target_mean,target_std) || Normal(online_mean,target_std))
                kl_mean = torch.distributions.kl.kl_divergence(target_base_per_dim, online_mean_dist.base_dist)  # (2B,A)
                mean_kl_mean = kl_mean.mean(dim=0)  # (A,)
                loss_kl_mean = torch.sum(alpha_mean.detach() * mean_kl_mean)
                loss_alpha_mean = torch.sum(alpha_mean * (self.eps_mu - mean_kl_mean.detach()))

                # --- std regression (fix mean to target mean) ---
                online_std_dist = torch.distributions.Independent(torch.distributions.Normal(t_mean, o_std), 1)
                logp_std = online_std_dist.log_prob(sampled_actions)  # (N,2B)

                loss_pg_std = -(logp_std * impr_distr).sum(dim=0).mean()

                kl_std = torch.distributions.kl.kl_divergence(target_base_per_dim, online_std_dist.base_dist)  # (2B,A)
                mean_kl_std = kl_std.mean(dim=0)
                loss_kl_std = torch.sum(alpha_std.detach() * mean_kl_std)
                loss_alpha_std = torch.sum(alpha_std * (self.eps_sigma - mean_kl_std.detach()))

                actor_loss = loss_pg_mean + loss_pg_std + loss_kl_mean + loss_kl_std

            # actor step
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
            self.actor_optimizer.step()

            # dual step
            dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta
            self.dual_optimizer.zero_grad(set_to_none=True)
            dual_loss.backward()
            dual_grad_norm = torch.nn.utils.clip_grad_norm_(self.duals.parameters(), self.grad_norm_clip)
            self.dual_optimizer.step()

            # clamp dual parameters (as in CleanRL)
            with torch.no_grad():
                self.duals.log_eta.data.clamp_(min=_MIN_LOG_TEMPERATURE)
                self.duals.log_alpha_mean.data.clamp_(min=_MIN_LOG_ALPHA)
                self.duals.log_alpha_stddev.data.clamp_(min=_MIN_LOG_ALPHA)
                self.duals.log_penalty_temperature.data.clamp_(min=_MIN_LOG_TEMPERATURE)

            return (
                q_loss, actor_loss, dual_loss,
                old_q.mean(),
                eta.detach(),
                penalty_temperature.detach(),
                alpha_mean.detach().mean(),
                alpha_std.detach().mean(),
                loss_eta.detach(),
                loss_alpha_mean.detach() + loss_alpha_std.detach(),
                mean_kl_mean.detach().mean(),
                mean_kl_std.detach().mean(),
                actor_grad_norm.detach(),
                critic_grad_norm.detach(),
                dual_grad_norm.detach(),
                o_std.detach().min(dim=1).values.mean(),
                o_std.detach().max(dim=1).values.mean(),
            )

        self.set_train_mode()

        state_t0 = None  # just for clarity
        prev_loop_end = time.time()

        while global_step < self.total_timesteps:
            start_time = time.time()
            torch.compiler.cudagraph_mark_step_begin()
            if logging_time_prev is not None:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)

            # -----------------------
            # Acting
            # -----------------------
            if global_step < self.learning_starts:
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)], dtype=np.float32)
                # normalize to [-1,1] for critic/policy space
                action_norm = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
                action_norm_t = torch.as_tensor(action_norm, dtype=torch.float32, device=self.device)
                processed_action_t = torch.as_tensor(processed_action, dtype=torch.float32, device=self.device)
            else:
                obs_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    action_norm_t, processed_action_t = self.env_actor.sample_action(obs_t)

                # env step uses numpy
                processed_action = processed_action_t.detach().cpu().numpy()

            next_state, reward, terminated, truncated, info = self.train_env.step(processed_action)
            done = terminated | truncated

            actual_next_state = next_state.copy()
            dones_this_rollout = 0
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1

            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)

            # store to replay buffer (Torch tensors, on device)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            next_state_t = torch.as_tensor(actual_next_state, dtype=torch.float32, device=self.device)
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            terminated_t = torch.as_tensor(terminated.astype(np.float32), dtype=torch.float32, device=self.device)
            truncated_t = torch.as_tensor(truncated.astype(np.float32), dtype=torch.float32, device=self.device)

            replay_buffer.add(state_t, next_state_t, action_norm_t, reward_t, terminated_t, truncated_t)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)

            # -----------------------
            # Decisions this step
            # -----------------------
            should_learning_start = global_step > self.learning_starts
            # update every `optimize_every` env steps per env => every optimize_every loop iterations
            loop_idx = global_step // self.nr_envs
            should_optimize = should_learning_start and (loop_idx % self.optimize_every == 0)

            should_try_to_save = should_learning_start and self.save_model and (dones_this_rollout > 0)
            should_log = global_step >= next_log_step
            should_evaluate = (next_eval_step is not None) and (global_step >= next_eval_step)

            # env_actor update schedule
            if global_step >= next_env_actor_update_step:
                self.env_actor.load_state_dict(self.target_actor.state_dict())
                next_env_actor_update_step += self.variable_update_period

            # -----------------------
            # Optimization
            # -----------------------
            if should_optimize:
                batch = replay_buffer.sample(self.batch_size)
                (b_s, b_ns, b_a, b_r, b_d, b_tr, b_k) = batch

                (
                    q_loss, actor_loss, dual_loss,
                    q_mean,
                    eta, pen_temp, alpha_mean_m, alpha_std_m,
                    loss_eta, loss_alpha,
                    kl_mean_avg, kl_std_avg,
                    actor_gn, critic_gn, dual_gn,
                    std_min, std_max,
                ) = train_step(b_s, b_ns, b_a, b_r, b_d, b_tr, b_k)

                sgd_steps += 1
                nr_updates += 1

                # target network update
                if sgd_steps % self.target_network_update_period == 0:
                    self.target_actor.load_state_dict(self.actor.state_dict())
                    self.critic.q_target.load_state_dict(self.critic.q.state_dict())

                # schedulers
                if self.anneal_learning_rate:
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()
                    self.dual_scheduler.step()

                # collect metrics
                lr_val = self.policy_q_lr if not self.anneal_learning_rate else self.actor_scheduler.get_last_lr()[0]
                optimization_metrics = {
                    "loss/q_loss": float(q_loss.item()),
                    "loss/actor_loss": float(actor_loss.item()),
                    "loss/dual_loss": float(dual_loss.item()),
                    "loss/loss_eta": float(loss_eta.item()),
                    "loss/loss_alpha": float(loss_alpha.item()),
                    "q_value/q_mean": float(q_mean.item()),
                    "duals/log_eta": float(self.duals.log_eta.detach().item()),
                    "duals/eta": float(eta.item()),
                    "duals/log_penalty_temperature": float(self.duals.log_penalty_temperature.detach().item()),
                    "duals/penalty_temperature": float(pen_temp.item()),
                    "duals/alpha_mean_mean": float(alpha_mean_m.item()),
                    "duals/alpha_std_mean": float(alpha_std_m.item()),
                    "kl/mean_kl_avg": float(kl_mean_avg.item()),
                    "kl/std_kl_avg": float(kl_std_avg.item()),
                    "gradients/actor_grad_norm": float(actor_gn.item()),
                    "gradients/critic_grad_norm": float(critic_gn.item()),
                    "gradients/dual_grad_norm": float(dual_gn.item()),
                    "policy/std_min_mean": float(std_min.item()),
                    "policy/std_max_mean": float(std_max.item()),
                    "lr/policy_q_lr": float(lr_val),
                    "steps/sgd_steps": float(sgd_steps),
                }
                for k, v in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(k, []).append(v)

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)

            # -----------------------
            # Evaluation
            # -----------------------
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                        act = self.actor.get_deterministic_action(torch.as_tensor(eval_state, dtype=torch.float32, device=self.device))
                    eval_state, _, eval_term, eval_trunc, eval_info = self.eval_env.step(act.detach().cpu().numpy())
                    eval_done = eval_term | eval_trunc
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(
                                self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i)
                            )
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(
                                self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i)
                            )
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                self.set_train_mode()
                next_eval_step += self.evaluation_frequency

            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)

            # -----------------------
            # Saving
            # -----------------------
            if should_try_to_save:
                mean_return = float(np.mean(saving_return_buffer)) if len(saving_return_buffer) > 0 else -np.inf
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            saving_end_time = time.time()
            if prev_saving_end_time is not None:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            # -----------------------
            # Logging
            # -----------------------
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_updates"] = nr_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes

                rollout_info_metrics = {}
                env_info_metrics = {}
                if step_info_collection:
                    for info_name, values in step_info_collection.items():
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = float(np.mean(values))
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value

                time_metrics = {k: float(np.mean(v)) for k, v in time_metrics_collection.items()}
                opt_metrics = {k: float(np.mean(v)) for k, v in optimization_metrics_collection.items()}
                eval_metrics = {k: float(np.mean(v)) for k, v in evaluation_metrics_collection.items()}
                combined = {**rollout_info_metrics, **eval_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **opt_metrics}
                for k, v in combined.items():
                    self.log(k, v, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

                self.end_logging()
                next_log_step += self.logging_frequency

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
        file_path = os.path.join(self.save_path, "best.model")
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
        }
        torch.save(save_dict, file_path)
        if self.track_wandb:
            wandb.save(file_path, base_path=os.path.dirname(file_path))

    @staticmethod
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
        return model

    def test(self, episodes):
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0.0
            state, _ = self.eval_env.reset()
            while not np.all(done):
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.bf16_mixed_precision_training):
                    action = self.actor.get_deterministic_action(torch.as_tensor(state, dtype=torch.float32, device=self.device))
                state, reward, terminated, truncated, _ = self.eval_env.step(action.detach().cpu().numpy())
                done = terminated | truncated
                episode_return += float(np.mean(reward))
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

    @staticmethod
    def general_properties():
        return GeneralProperties
