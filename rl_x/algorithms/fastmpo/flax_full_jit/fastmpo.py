import os
import shutil
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.fastmpo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.fastmpo.flax_full_jit.policy import get_policy
from rl_x.algorithms.fastmpo.flax_full_jit.critic import get_critic
from rl_x.algorithms.fastmpo.flax_full_jit.dual_variables import DualVariables
from rl_x.algorithms.fastmpo.flax_full_jit.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class FastMPO:
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
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.dual_critic = config.algorithm.dual_critic
        self.action_clipping = config.algorithm.action_clipping
        self.policy_learning_rate = config.algorithm.policy_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_policy_learning_rate = config.algorithm.anneal_policy_learning_rate
        self.anneal_critic_learning_rate = config.algorithm.anneal_critic_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.policy_weight_decay = config.algorithm.policy_weight_decay
        self.critic_weight_decay = config.algorithm.critic_weight_decay
        self.dual_weight_decay = config.algorithm.dual_weight_decay
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.collect_data_with_online_policy = config.algorithm.collect_data_with_online_policy
        self.action_sampling_number = config.algorithm.action_sampling_number
        self.epsilon_non_parametric = config.algorithm.epsilon_non_parametric
        self.epsilon_parametric_mu = config.algorithm.epsilon_parametric_mu
        self.epsilon_parametric_sigma = config.algorithm.epsilon_parametric_sigma
        self.epsilon_penalty = config.algorithm.epsilon_penalty
        self.init_log_eta = config.algorithm.init_log_eta
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.init_log_penalty_temperature = config.algorithm.init_log_penalty_temperature
        self.float_epsilon = config.algorithm.float_epsilon
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.critic_tau = config.algorithm.critic_tau
        self.policy_tau = config.algorithm.policy_tau
        self.gamma = config.algorithm.gamma
        self.nr_atoms = config.algorithm.nr_atoms
        self.n_steps = config.algorithm.n_steps
        self.clipped_double_q_learning = config.algorithm.clipped_double_q_learning
        self.nr_critic_updates_per_policy_update = config.algorithm.nr_critic_updates_per_policy_update
        self.nr_policy_updates_per_step = config.algorithm.nr_policy_updates_per_step
        self.enable_observation_normalization = config.algorithm.enable_observation_normalization
        self.normalizer_epsilon = config.algorithm.normalizer_epsilon
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.nr_envs * (self.total_timesteps // self.nr_envs)
        self.nr_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_loggings_per_eval_save_iteration = self.evaluation_and_save_frequency // self.logging_frequency
        self.nr_updates_per_logging_iteration = self.logging_frequency // self.nr_envs
        self.nr_critic_updates_per_step = self.nr_policy_updates_per_step * self.nr_critic_updates_per_policy_update
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        if self.evaluation_and_save_frequency % self.nr_envs != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of nr envs.")
        
        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("Logging frequency must be a multiple of nr envs.")
        
        if self.evaluation_and_save_frequency % self.logging_frequency != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of logging frequency.")
        
        if self.learning_starts < self.n_steps:
            raise ValueError("The replay buffer must at least be filled with n_steps transitions before learning starts.")
        
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, dual_key, reset_key = jax.random.split(self.key, 5)
        reset_key = jax.random.split(reset_key, self.nr_envs)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        nr_actions = np.prod(self.train_env.single_action_space.shape).item()
        self.dual_variables = DualVariables(nr_actions, self.init_log_eta, self.init_log_alpha_mean, self.init_log_alpha_stddev, self.init_log_penalty_temperature)

        def policy_linear_schedule(count):
            step = count * self.nr_envs
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.policy_learning_rate * fraction
        
        def critic_linear_schedule(count):
            step = count * self.nr_envs
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.critic_learning_rate * fraction
        
        def dual_linear_schedule(count):
            step = count * self.nr_envs
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.dual_learning_rate * fraction
        
        self.policy_learning_rate = policy_linear_schedule if self.anneal_policy_learning_rate else self.policy_learning_rate
        self.critic_learning_rate = critic_linear_schedule if self.anneal_critic_learning_rate else self.critic_learning_rate
        self.dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate

        env_state = self.train_env.reset(reset_key, False)

        dummy_observation = env_state.next_observation
        dummy_action = jnp.zeros((self.nr_envs,) + self.as_shape, dtype=jnp.float32)

        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_observation),
            target_params=self.policy.init(policy_key, dummy_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.policy_learning_rate, weight_decay=self.policy_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2),
            )
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dummy_observation, dummy_action),
            target_params=self.critic.init(critic_key, dummy_observation, dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.critic_learning_rate, weight_decay=self.critic_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2),
            )
        )

        self.dual_variables_state = TrainState.create(
            apply_fn=self.dual_variables.apply,
            params=self.dual_variables.init(dual_key),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.dual_learning_rate, weight_decay=self.dual_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2),
            )
        )
        
        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": jnp.zeros((1, self.os_shape[0])),
                "running_var": jnp.ones((1, self.os_shape[0])),
                "running_std_dev": jnp.ones((1, self.os_shape[0])),
                "count": jnp.zeros(())
            }
        else:
            self.observation_normalizer_state = {}

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

 
    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key, 2)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state
            dual_variables_state = self.dual_variables_state
            observation_normalizer_state = self.observation_normalizer_state

            # Replay buffer
            states_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.os_shape[0]), dtype=jnp.float32)
            next_states_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.os_shape[0]), dtype=jnp.float32)
            actions_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs, self.as_shape[0]), dtype=jnp.float32)
            rewards_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32)
            dones_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32)
            truncations_buffer = jnp.zeros((self.buffer_size_per_env, self.nr_envs), dtype=jnp.float32)
            replay_buffer = {
                "states": states_buffer,
                "next_states": next_states_buffer,
                "actions": actions_buffer,
                "rewards": rewards_buffer,
                "dones": dones_buffer,
                "truncations": truncations_buffer,
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32)
            }

            # Fill replay buffer until learning_starts
            def fill_replay_buffer(carry, _):
                policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = carry

                # Acting
                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                if self.enable_observation_normalization:
                    normalized_observation = (observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                else:
                    normalized_observation = observation
                policy_params = policy_state.params if self.collect_data_with_online_policy else policy_state.target_params
                action_mean, action_std = self.policy.apply(policy_params, normalized_observation)
                action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                processed_action = self.get_processed_action(action)
                env_state = self.train_env.step(env_state, processed_action)
                dones = env_state.terminated | env_state.truncated

                # Adding to replay buffer
                replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                replay_buffer["dones"] = replay_buffer["dones"].at[replay_buffer["pos"]].set(dones)
                replay_buffer["truncations"] = replay_buffer["truncations"].at[replay_buffer["pos"]].set(env_state.truncated)
                replay_buffer["pos"] = (replay_buffer["pos"] + 1) % self.buffer_size_per_env
                replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, self.buffer_size_per_env)

                if self.render:
                    def render(env_state):
                        return self.train_env.render(env_state)
                    
                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key), None
            
            key, subkey = jax.random.split(key)
            fill_replay_buffer_carry, _ = jax.lax.scan(fill_replay_buffer, (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, subkey), jnp.arange(self.learning_starts))
            policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = fill_replay_buffer_carry


            # Training
            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, learning_iteration_step):
                        policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = learning_iteration_carry

                        # Acting
                        key, subkey = jax.random.split(key)
                        observation = env_state.next_observation
                        if self.enable_observation_normalization:
                            normalized_observation = (observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                        else:
                            normalized_observation = observation
                        policy_params = policy_state.params if self.collect_data_with_online_policy else policy_state.target_params
                        action_mean, action_std = self.policy.apply(policy_params, normalized_observation)
                        action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
                        processed_action = self.get_processed_action(action)
                        env_state = self.train_env.step(env_state, processed_action)
                        dones = env_state.terminated | env_state.truncated

                        # Adding to replay buffer
                        replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                        replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                        replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                        replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                        replay_buffer["dones"] = replay_buffer["dones"].at[replay_buffer["pos"]].set(dones)
                        replay_buffer["truncations"] = replay_buffer["truncations"].at[replay_buffer["pos"]].set(env_state.truncated)
                        replay_buffer["pos"] = (replay_buffer["pos"] + 1) % self.buffer_size_per_env
                        replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, self.buffer_size_per_env)

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)
                            
                            env_state = jax.experimental.io_callback(render, env_state, env_state)


                        # Optimizing - Critic and Policy
                        def critic_loss_fn(policy_target_params, critic_params, critic_target_params, normalized_state, normalized_next_state, action, reward, done, truncated, effective_n_steps, key):
                            # Critic loss
                            next_action_mean, next_action_std = self.policy.apply(policy_target_params, normalized_next_state)
                            sampled_next_actions_raw = next_action_mean[None, :] + next_action_std[None, :] * jax.random.normal(key, shape=(self.action_sampling_number,) + next_action_mean.shape)
                            sampled_next_actions = jnp.clip(sampled_next_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_next_actions_raw

                            expanded_next_states = jnp.repeat(normalized_next_state[None, :], self.action_sampling_number, axis=0)
                            target_next_logits = self.critic.apply(critic_target_params, expanded_next_states, sampled_next_actions) # (n_critics, N, nr_atoms)
                            target_next_pmf = jax.nn.softmax(target_next_logits, axis=-1) # (n_critics, K, nr_atoms)
                            mean_target_next_pmf = jnp.mean(target_next_pmf, axis=1) # (n_critics, nr_atoms)

                            delta_z = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                            q_support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
                            bootstrap = 1.0 - (done * (1.0 - truncated))
                            discount = (self.gamma ** effective_n_steps) * bootstrap
                            target_z = jnp.clip(reward + discount * q_support, self.v_min, self.v_max)
                            
                            b = (target_z - self.v_min) / delta_z
                            l = jnp.floor(b).astype(jnp.int32)
                            u = jnp.ceil(b).astype(jnp.int32)

                            is_int = (u == l)
                            l_mask = is_int & (l > 0)
                            u_mask = is_int & (l == 0)

                            l = jnp.where(l_mask, l - 1, l)
                            u = jnp.where(u_mask, u + 1, u)

                            wt_l = (u.astype(jnp.float32) - b)
                            wt_u = (b - l.astype(jnp.float32))

                            n_critics = mean_target_next_pmf.shape[0]
                            proj = jnp.zeros_like(mean_target_next_pmf)

                            critic_idxs = jnp.arange(n_critics)[:, None]
                            critic_idxs = jnp.repeat(critic_idxs, self.nr_atoms, axis=1)
                            l_idxs = jnp.repeat(l[None, :], n_critics, axis=0)
                            u_idxs = jnp.repeat(u[None, :], n_critics, axis=0)

                            proj = proj.at[(critic_idxs, l_idxs)].add(mean_target_next_pmf * wt_l)
                            proj = proj.at[(critic_idxs, u_idxs)].add(mean_target_next_pmf * wt_u)

                            qf_next_target_value = jnp.sum(proj * q_support, axis=-1)  # (n_critics,)

                            if self.dual_critic and self.clipped_double_q_learning:
                                use_first = qf_next_target_value[0] <= qf_next_target_value[1]
                                chosen = jnp.where(use_first, proj[0], proj[1])  # (nr_atoms,)
                                target_dist = jnp.stack([chosen, chosen], axis=0)  # (2, nr_atoms)
                            else:
                                target_dist = proj  # (n_critics, nr_atoms)
                            
                            action_in = jnp.clip(action, -1.0, 1.0) if self.action_clipping else action
                            current_logits = self.critic.apply(critic_params, normalized_state, action_in)  # (n_critics, nr_atoms)
                            current_log_pmf = jax.nn.log_softmax(current_logits, axis=-1)
                            q_loss_per_critic = -jnp.sum(target_dist * current_log_pmf, axis=-1)  # (n_critics,)
                            q_loss = jnp.sum(q_loss_per_critic)

                            # Create metrics
                            metrics = {
                                "loss/q_loss": q_loss,
                                "q/q_mean": jnp.mean(qf_next_target_value),
                                "q/q_max": jnp.max(qf_next_target_value),
                                "q/q_min": jnp.min(qf_next_target_value),
                            }

                            return q_loss, (metrics)
                        
                        def policy_and_dual_loss_fn(policy_params, policy_target_params, critic_target_params, dual_variables_params, normalized_state, normalized_next_state, key1, key2):   
                            # Policy and dual loss
                            stacked_states = jnp.concatenate([normalized_state[None, :], normalized_next_state[None, :]], axis=0)  # (2, obs)
                            target_action_mean, target_action_std = self.policy.apply(policy_target_params, stacked_states)
                            sampled_actions_raw = target_action_mean[None, :, :] + target_action_std[None, :, :] * jax.random.normal(key2, shape=(self.action_sampling_number,) + target_action_mean.shape)  # (sampled actions, 2, action_dim)
                            sampled_actions = jnp.clip(sampled_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_actions_raw

                            expanded_states = jnp.repeat(stacked_states[None, :, :], self.action_sampling_number, axis=0)  # (K,2,obs)
                            flat_states = expanded_states.reshape((-1, expanded_states.shape[-1]))      # (K*2, obs)
                            flat_actions = sampled_actions.reshape((-1, sampled_actions.shape[-1]))    # (K*2, act)
                            logits = self.critic.apply(critic_target_params, flat_states, flat_actions)  # (n_critics, K*2, nr_atoms)
                            pmf = jax.nn.softmax(logits, axis=-1)
                            q_support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
                            q_vals = jnp.sum(pmf * q_support, axis=-1)
                            q_vals = q_vals.reshape((q_vals.shape[0], self.action_sampling_number, 2))  # (n_critics,K,2)
                            if q_vals.shape[0] == 1:
                                q_vals = q_vals[0]
                            if self.clipped_double_q_learning:
                                q_vals = jnp.min(q_vals, axis=0)
                            else:
                                q_vals =  jnp.mean(q_vals, axis=0)

                            log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature = self.dual_variables.apply(dual_variables_params)
                            eta = jax.nn.softplus(log_eta) + self.float_epsilon
                            eta_s = eta[0]
                            improvement_dist = jax.nn.softmax(q_vals / stop_gradient(eta_s), axis=0)  # (K,2)

                            q_logsumexp = jax.scipy.special.logsumexp(q_vals / eta_s, axis=0)  # (2,)
                            loss_eta = eta_s * (self.epsilon_non_parametric + jnp.mean(q_logsumexp) - jnp.log(self.action_sampling_number))

                            if self.action_clipping:
                                penalty_temperature = jax.nn.softplus(log_penalty_temperature)[0] + self.float_epsilon
                                diff_oob = sampled_actions_raw - jnp.clip(sampled_actions_raw, -1.0, 1.0)  # (K,2,act)
                                cost_oob = -jnp.linalg.norm(diff_oob, axis=-1)  # (K,2)
                                penalty_improvement = jax.nn.softmax(cost_oob / stop_gradient(penalty_temperature), axis=0)  # (K,2)

                                penalty_logsumexp = jax.scipy.special.logsumexp(cost_oob / penalty_temperature, axis=0)  # (2,)
                                loss_penalty_temp = penalty_temperature * (self.epsilon_penalty + jnp.mean(penalty_logsumexp) - jnp.log(self.action_sampling_number))

                                improvement_dist = improvement_dist + penalty_improvement
                                loss_eta = loss_eta + loss_penalty_temp
                            else:
                                penalty_temperature = jnp.array(0.0, dtype=jnp.float32)

                            online_action_mean, online_action_std =  self.policy.apply(policy_params, stacked_states)  # (2,act)

                            alpha_mean = jax.nn.softplus(log_alpha_mean) + self.float_epsilon
                            alpha_std = jax.nn.softplus(log_alpha_stddev) + self.float_epsilon

                            logprob_mean = jnp.sum(-0.5 * ((((sampled_actions_raw - online_action_mean) / target_action_std) ** 2) + jnp.log(2.0 * jnp.pi)) - jnp.log(target_action_std), axis=-1)  # (sampled actions, 2 * batch)

                            loss_pg_mean = -(logprob_mean * improvement_dist).sum(axis=0).mean()

                            kl_mean_std0 = jnp.clip(target_action_std, min=self.float_epsilon)
                            kl_mean_std1 = jnp.clip(target_action_std, min=self.float_epsilon)
                            kl_mean_var0 = kl_mean_std0 ** 2
                            kl_mean_var1 = kl_mean_std1 ** 2
                            kl_mean = jnp.log(kl_mean_std1 / kl_mean_std0) + (kl_mean_var0 + (target_action_mean - online_action_mean) ** 2) / (2.0 * kl_mean_var1) - 0.5
                            mean_kl_mean = kl_mean.mean(axis=0)  # (action_dim,)
                            loss_kl_mean = jnp.sum(stop_gradient(alpha_mean) * mean_kl_mean)
                            loss_alpha_mean = jnp.sum(alpha_mean * (self.epsilon_parametric_mu - stop_gradient(mean_kl_mean)))

                            logprob_std = jnp.sum(-0.5 * ((((sampled_actions - target_action_mean) / online_action_std) ** 2) + jnp.log(2.0 * jnp.pi)) - jnp.log(online_action_std), axis=-1)  # (sampled actions, 2 * batch)

                            loss_pg_std = -(logprob_std * improvement_dist).sum(axis=0).mean()

                            kl_std_std0 = jnp.clip(target_action_std, min=self.float_epsilon)
                            kl_std_std1 = jnp.clip(online_action_std, min=self.float_epsilon)
                            kl_std_var0 = kl_std_std0 ** 2
                            kl_std_var1 = kl_std_std1 ** 2
                            kl_std = jnp.log(kl_std_std1 / kl_std_std0) + (kl_std_var0 + (target_action_mean - target_action_mean) ** 2) / (2.0 * kl_std_var1) - 0.5
                            mean_kl_std = kl_std.mean(axis=0)
                            loss_kl_std = jnp.sum(stop_gradient(alpha_std) * mean_kl_std)
                            loss_alpha_std = jnp.sum(alpha_std * (self.epsilon_parametric_sigma - stop_gradient(mean_kl_std)))

                            actor_loss = loss_pg_mean + loss_pg_std + loss_kl_mean + loss_kl_std
                            dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta
                            loss = actor_loss + dual_loss

                            # Create metrics
                            metrics = {
                                "loss/actor_loss": actor_loss,
                                "loss/loss_pg_mean": loss_pg_mean,
                                "loss/loss_pg_std": loss_pg_std,
                                "loss/loss_kl_mean": loss_kl_mean,
                                "loss/loss_kl_std": loss_kl_std,
                                "loss/dual_loss": dual_loss,
                                "loss/loss_alpha_mean": loss_alpha_mean,
                                "loss/loss_alpha_std": loss_alpha_std,
                                "loss/loss_eta": loss_eta,
                                "dual/eta": eta_s,
                                "dual/penalty_temperature": penalty_temperature,
                                "dual/alpha_mean": jnp.mean(alpha_mean),
                                "dual/alpha_std": jnp.mean(alpha_std),
                                "kl/mean_kl_mean": jnp.mean(mean_kl_mean),
                                "kl/mean_kl_std": jnp.mean(mean_kl_std),
                                "q/improvement_q_mean": jnp.mean(q_vals),
                                "policy/std_min_mean": jnp.mean(jnp.min(online_action_std, axis=-1)),
                                "policy/std_max_mean": jnp.mean(jnp.max(online_action_std, axis=-1)),
                            }

                            return loss, (metrics)
                        

                        vmap_critic_loss_fn = jax.vmap(critic_loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                        safe_mean = lambda x: jnp.mean(x) if x is not None else x
                        mean_vmapped_critic_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_critic_loss_fn(*a, **k))
                        grad_critic_loss_fn = jax.value_and_grad(mean_vmapped_critic_loss_fn, argnums=(1,), has_aux=True)

                        vmap_policy_and_dual_loss_fn = jax.vmap(policy_and_dual_loss_fn, in_axes=(None, None, None, None, 0, 0, 0, 0), out_axes=0)
                        mean_vmapped_policy_and_dual_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_policy_and_dual_loss_fn(*a, **k))
                        grad_policy_and_dual_loss_fn = jax.value_and_grad(mean_vmapped_policy_and_dual_loss_fn, argnums=(0, 3), has_aux=True)

                        # Sample batch from replay buffer and handling n-step returns
                        key, idx_key_t, idx_key_e, noise_key = jax.random.split(key, 4)

                        def full_case(_):
                            last_idx = (replay_buffer["pos"] - 1) % self.buffer_size_per_env
                            last_trunc_row = replay_buffer["truncations"][last_idx]
                            last_done_row = replay_buffer["dones"][last_idx]
                            patched_last_trunc_row = jnp.where(last_done_row > 0.0, last_trunc_row, jnp.ones_like(last_trunc_row))
                            trunc_patched = replay_buffer["truncations"].at[last_idx].set(patched_last_trunc_row)
                            return self.buffer_size_per_env, trunc_patched

                        def not_full_case(_):
                            max_start = jnp.maximum(1, replay_buffer["size"] - self.n_steps + 1)
                            return max_start, replay_buffer["truncations"]

                        max_start, truncations_for_sampling = jax.lax.cond(replay_buffer["size"] >= self.buffer_size_per_env, full_case,  not_full_case, operand=None)

                        idx1 = jax.random.randint(idx_key_t, (self.nr_critic_updates_per_step, self.batch_size), 0, max_start)
                        idx2 = jax.random.randint(idx_key_e, (self.nr_critic_updates_per_step, self.batch_size), 0, self.nr_envs)
                        update_keys = jax.random.split(noise_key, self.nr_critic_updates_per_step * self.batch_size * 3)
                        update_keys = update_keys.reshape(self.nr_critic_updates_per_step, self.batch_size, 3, -1)
                        states_all = replay_buffer["states"][idx1, idx2]
                        actions_all = replay_buffer["actions"][idx1, idx2]

                        steps = jnp.arange(self.n_steps)
                        all_indices = (idx1[..., None] + steps) % self.buffer_size_per_env  # (nr_critic_updates_per_step, batch_size, n_steps)
                        env_indices = jnp.broadcast_to(idx2[..., None], all_indices.shape)  # (nr_critic_updates_per_step, batch_size, n_steps)
                        flat_t = all_indices.reshape(-1)
                        flat_e = env_indices.reshape(-1)
                        all_rewards = replay_buffer["rewards"][flat_t, flat_e].reshape(all_indices.shape)
                        all_dones = replay_buffer["dones"][flat_t, flat_e].reshape(all_indices.shape)
                        all_truncations = truncations_for_sampling[flat_t, flat_e].reshape(all_indices.shape)

                        zeros_first = jnp.zeros((*all_dones.shape[:-1], 1), dtype=all_dones.dtype)
                        all_dones_shifted = jnp.concatenate([zeros_first, all_dones[..., :-1]], axis=-1)
                        done_masks = jnp.cumprod(1 - all_dones_shifted.astype(jnp.float32), axis=-1)
                        effective_n_steps_all = jnp.sum(done_masks, axis=-1)
                        discounts = self.gamma ** jnp.arange(self.n_steps)
                        rewards_all = jnp.sum(all_rewards * done_masks * discounts, axis=-1)

                        all_dones_int = (all_dones > 0.0).astype(jnp.int32)
                        all_trunc_int = (all_truncations > 0.0).astype(jnp.int32)
                        first_done = jnp.argmax(all_dones_int, axis=-1)
                        first_trunc = jnp.argmax(all_trunc_int, axis=-1)
                        no_dones = jnp.sum(all_dones_int , axis=-1) == 0
                        no_truncs = jnp.sum(all_trunc_int, axis=-1) == 0
                        first_done = jnp.where(no_dones, self.n_steps - 1, first_done)
                        first_trunc = jnp.where(no_truncs, self.n_steps - 1, first_trunc)
                        final_offset = jnp.minimum(first_done, first_trunc)
                        final_time_indices = jnp.take_along_axis(all_indices, final_offset[..., None], axis=-1).squeeze(-1)
                        flat_t_final = final_time_indices.reshape(-1)
                        flat_e_final = idx2.reshape(-1)
                        next_states_all = replay_buffer["next_states"][flat_t_final, flat_e_final].reshape(idx1.shape + (self.os_shape[0],))
                        dones_all = replay_buffer["dones"][flat_t_final, flat_e_final].reshape(idx1.shape)
                        truncations_all = truncations_for_sampling[flat_t_final, flat_e_final].reshape(idx1.shape)

                        if self.enable_observation_normalization:
                            combined_states = jnp.concatenate([states_all.reshape(-1, self.os_shape[0]), next_states_all.reshape(-1, self.os_shape[0])], axis=0)
                            batch_mean = jnp.mean(combined_states, axis=0, keepdims=True)
                            batch_var = jnp.var(combined_states, axis=0, keepdims=True)
                            batch_count = combined_states.shape[0]
                            new_count = observation_normalizer_state["count"] + batch_count
                            delta = batch_mean - observation_normalizer_state["running_mean"]
                            observation_normalizer_state["running_mean"] += delta * batch_count / new_count
                            delta2 = batch_mean - observation_normalizer_state["running_mean"]
                            m_a = observation_normalizer_state["running_var"] * observation_normalizer_state["count"]
                            m_b = batch_var * batch_count
                            m2 = m_a + m_b + jnp.square(delta2) * observation_normalizer_state["count"] * batch_count / new_count
                            observation_normalizer_state["running_var"] = m2 / new_count
                            observation_normalizer_state["running_std_dev"] = jnp.sqrt(observation_normalizer_state["running_var"])
                            observation_normalizer_state["count"] = new_count

                            normalized_states_all = (states_all - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                            normalized_next_states_all = (next_states_all - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                        else:
                            normalized_states_all = states_all
                            normalized_next_states_all = next_states_all

                        update_idx = 0
                        for _ in range(self.nr_policy_updates_per_step):
                            for _ in range(self.nr_critic_updates_per_policy_update):
                                normalized_states = normalized_states_all[update_idx]
                                normalized_next_states = normalized_next_states_all[update_idx]
                                actions = actions_all[update_idx]
                                rewards = rewards_all[update_idx]
                                dones = dones_all[update_idx]
                                truncations = truncations_all[update_idx]
                                effective_n_steps = effective_n_steps_all[update_idx]
                                keys_for_critic_update = update_keys[update_idx, :, 0, :]
                                keys1_for_policy_update = update_keys[update_idx, :, 1, :]
                                keys2_for_policy_update = update_keys[update_idx, :, 2, :]

                                (loss, (critic_metrics)), (critic_gradients,) = grad_critic_loss_fn(
                                    policy_state.target_params, critic_state.params, critic_state.target_params,
                                    normalized_states, normalized_next_states, actions, rewards, dones, truncations, effective_n_steps,
                                    keys_for_critic_update)

                                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                                critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.critic_tau))

                                critic_metrics["lr/critic_learning_rate"] = critic_state.opt_state[1].hyperparams["learning_rate"]
                                critic_metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                                update_idx += 1

                            normalized_states_for_policy = normalized_states
                            normalized_next_states_for_policy = normalized_next_states
                            
                            (loss, (policy_metrics)), (policy_gradients, dual_variables_gradients) = grad_policy_and_dual_loss_fn(
                                policy_state.params, policy_state.target_params, critic_state.target_params, dual_variables_state.params, 
                                normalized_states_for_policy, normalized_next_states_for_policy, keys1_for_policy_update, keys2_for_policy_update
                            )

                            policy_state = policy_state.apply_gradients(grads=policy_gradients)
                            dual_variables_state = dual_variables_state.apply_gradients(grads=dual_variables_gradients)

                            dual_variables_state = dual_variables_state.replace(
                                params={
                                    "params": {
                                        "log_eta": jnp.maximum(dual_variables_state.params["params"]["log_eta"], self.min_log_temperature),
                                        "log_alpha_mean": jnp.maximum(dual_variables_state.params["params"]["log_alpha_mean"], self.min_log_alpha),
                                        "log_alpha_stddev": jnp.maximum(dual_variables_state.params["params"]["log_alpha_stddev"], self.min_log_alpha),
                                        "log_penalty_temperature": dual_variables_state.params["params"]["log_penalty_temperature"],
                                    }
                                }
                            )
                            
                            policy_state = policy_state.replace(target_params=optax.incremental_update(policy_state.params, policy_state.target_params, self.policy_tau))

                            policy_metrics["lr/policy_learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                            policy_metrics["lr/dual_variables_learning_rate"] = dual_variables_state.opt_state[1].hyperparams["learning_rate"]
                            policy_metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                            policy_metrics["gradients/dual_variables_grad_norm"] = optax.global_norm(dual_variables_gradients)

                        metrics = {**critic_metrics, **policy_metrics}

                        return (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key), (env_state.info, metrics)
                        
                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(learning_iteration, (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, subkey), jnp.arange(self.nr_updates_per_logging_iteration))
                    policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = learning_iteration_carry
                    infos, optimization_metrics = info_and_optimization_metrics
                    infos = {key: jnp.mean(infos[key]) for key in infos}
                    optimization_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}


                    # Logging
                    nr_update_iteration = (eval_save_iteration_step * self.nr_loggings_per_eval_save_iteration * self.nr_updates_per_logging_iteration) + (logging_iteration_step+1) * self.nr_updates_per_logging_iteration
                    steps_metrics = {
                        "steps/nr_env_steps": nr_update_iteration * self.nr_envs,
                        "steps/nr_policy_updates": nr_update_iteration * self.nr_policy_updates_per_step,
                        "steps/nr_critic_updates": nr_update_iteration * self.nr_critic_updates_per_policy_update * self.nr_policy_updates_per_step,
                    }

                    combined_metrics = {**infos, **steps_metrics, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_envs * self.nr_updates_per_logging_iteration) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(metrics["steps/nr_env_steps"])
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, parallel_seed_id))

                    return (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(logging_iteration, (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, subkey), jnp.arange(self.nr_loggings_per_eval_save_iteration))
                policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key = logging_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        policy_state, eval_env_state = carry
                        if self.enable_observation_normalization:
                            eval_normalized_observation = (eval_env_state.next_observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                        else:
                            eval_normalized_observation = eval_env_state.next_observation
                        eval_action_mean, _ = self.policy.apply(policy_state.params, eval_normalized_observation)
                        eval_action = eval_action_mean
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)
                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    (policy_state, eval_env_state), _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def eval_callback(args):
                        metrics, global_step = args
                        global_step = int(global_step)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    global_step = (eval_save_iteration_step + 1) * self.evaluation_and_save_frequency
                    jax.debug.callback(eval_callback, (eval_metrics, global_step))
                

                # Saving
                if self.save_model:
                    def save_with_check(policy_state, critic_state, dual_variables_state, observation_normalizer_state):
                        self.save(policy_state, critic_state, dual_variables_state, observation_normalizer_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, dual_variables_state, observation_normalizer_state)

                
                return (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key), None

            jax.lax.scan(eval_save_iteration, (policy_state, critic_state, dual_variables_state, observation_normalizer_state, replay_buffer, env_state, key), jnp.arange(self.nr_eval_save_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")
    

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


    def save(self, policy_state, critic_state, dual_variables_state, observation_normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "dual_variables": dual_variables_state,
            "observation_normalizer": observation_normalizer_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)
    

    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"
        
        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = FastMPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "dual_variables": model.dual_variables_state,
            "observation_normalizer": model.observation_normalizer_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.dual_variables_state = checkpoint["dual_variables"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            # key, std_dev_key = jax.random.split(key)
            observation = env_state.next_observation
            if self.enable_observation_normalization:
                normalized_observation = (observation - self.observation_normalizer_state["running_mean"]) / (self.observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
            else:
                normalized_observation = observation
            action_mean, _ = self.policy.apply(self.policy_state.params, normalized_observation)
            action = action_mean
            processed_action = self.get_processed_action(action)
            env_state = self.eval_env.step(env_state, processed_action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.eval_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.eval_env.render(env_state)


    def general_properties():
        return GeneralProperties
