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
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.fasttd3.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.fasttd3.flax_full_jit.policy import get_policy
from rl_x.algorithms.fasttd3.flax_full_jit.critic import get_critic
from rl_x.algorithms.fasttd3.flax_full_jit.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class FastTD3:
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
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.weight_decay = config.algorithm.weight_decay
        self.batch_size = config.algorithm.batch_size
        self.buffer_size_per_env = config.algorithm.buffer_size_per_env
        self.learning_starts = config.algorithm.learning_starts
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.nr_atoms = config.algorithm.nr_atoms
        self.n_steps = config.algorithm.n_steps
        self.noise_std_min = config.algorithm.noise_std_min
        self.noise_std_max = config.algorithm.noise_std_max
        self.smoothing_epsilon = config.algorithm.smoothing_epsilon
        self.smoothing_clip_value = config.algorithm.smoothing_clip_value
        self.nr_critic_updates_per_policy_update = config.algorithm.nr_critic_updates_per_policy_update
        self.nr_policy_updates_per_step = config.algorithm.nr_policy_updates_per_step
        self.clipped_double_q_learning = config.algorithm.clipped_double_q_learning
        self.max_grad_norm = config.algorithm.max_grad_norm
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
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, self.nr_envs)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            step = count * self.nr_envs
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)

        if self.max_grad_norm != -1.0:
            policy_tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.policy_learning_rate, weight_decay=self.weight_decay),
            )
        else:
            policy_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=self.policy_learning_rate, weight_decay=self.weight_decay)
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=policy_tx
        )

        dummy_action = jnp.zeros((self.nr_envs,) + self.as_shape, dtype=jnp.float32)

        if self.max_grad_norm != -1.0:
            critic_tx = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adamw)(learning_rate=self.q_learning_rate, weight_decay=self.weight_decay),
            )
        else:
            critic_tx = optax.inject_hyperparams(optax.adamw)(learning_rate=self.q_learning_rate, weight_decay=self.weight_decay)
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, env_state.next_observation, dummy_action),
            target_params=self.critic.init(critic_key, env_state.next_observation, dummy_action),
            tx=critic_tx
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
            key, reset_key, noise_std_key = jax.random.split(key, 3)
            reset_keys = jax.random.split(reset_key, self.nr_envs)
            env_state = self.train_env.reset(reset_keys, False)

            policy_state = self.policy_state
            critic_state = self.critic_state
            observation_normalizer_state = self.observation_normalizer_state

            noise_scales = jax.random.uniform(noise_std_key, (self.nr_envs, 1), minval=self.noise_std_min, maxval=self.noise_std_max)

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
                policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = carry

                # Acting
                key, subkey = jax.random.split(key)
                observation = env_state.next_observation
                if self.enable_observation_normalization:
                    normalized_observation = (observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                else:
                    normalized_observation = observation
                action = self.policy.apply(policy_state.params, normalized_observation)
                action += jax.random.normal(subkey, action.shape) * noise_scales
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

                # Generate new noise scales for environments that are done
                noise_scales = jnp.where(
                    dones[:, None],
                    jax.random.uniform(key, (self.nr_envs, 1), minval=self.noise_std_min, maxval=self.noise_std_max),
                    noise_scales
                )

                if self.render:
                    def render(env_state):
                        return self.train_env.render(env_state)
                    
                    env_state = jax.experimental.io_callback(render, env_state, env_state)

                return (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key), None
            
            key, subkey = jax.random.split(key)
            fill_replay_buffer_carry, _ = jax.lax.scan(fill_replay_buffer, (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, subkey), jnp.arange(self.learning_starts))
            policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = fill_replay_buffer_carry


            # Training
            def eval_save_iteration(eval_save_iteration_carry, eval_save_iteration_step):
                policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = eval_save_iteration_carry

                def logging_iteration(logging_iteration_carry, logging_iteration_step):
                    policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = logging_iteration_carry

                    def learning_iteration(learning_iteration_carry, learning_iteration_step):
                        policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = learning_iteration_carry

                        # Acting
                        key, subkey1, subkey2 = jax.random.split(key, 3)
                        observation = env_state.next_observation
                        if self.enable_observation_normalization:
                            normalized_observation = (observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                        else:
                            normalized_observation = observation
                        action = self.policy.apply(policy_state.params, normalized_observation)
                        action += jax.random.normal(subkey1, action.shape) * noise_scales
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

                        # Generate new noise scales for environments that are done
                        noise_scales = jnp.where(
                            dones[:, None],
                            jax.random.uniform(subkey2, (self.nr_envs, 1), minval=self.noise_std_min, maxval=self.noise_std_max),
                            noise_scales
                        )

                        if self.render:
                            def render(env_state):
                                return self.train_env.render(env_state)
                            
                            env_state = jax.experimental.io_callback(render, env_state, env_state)


                        # Optimizing - Critic and Policy
                        def critic_loss_fn(policy_params, critic_params, critic_target_params, normalized_state, normalized_next_state, action, reward, done, truncated, effective_n_steps, key):
                            # Critic loss
                            clipped_noise = jnp.clip(jax.random.normal(key, action.shape) * self.smoothing_epsilon, -self.smoothing_clip_value, self.smoothing_clip_value)
                            next_action = jnp.clip(self.policy.apply(policy_params, normalized_next_state) + clipped_noise, -1.0, 1.0)

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

                            next_dist = jax.nn.softmax(self.critic.apply(critic_target_params, normalized_next_state, next_action)) # (2, nr_atoms) for the 2 critics
                            proj_dist = jnp.zeros_like(next_dist)
                            wt_l = (u.astype(jnp.float32) - b)
                            wt_u = (b - l.astype(jnp.float32))

                            n_critics = next_dist.shape[0]
                            critic_idxs = jnp.arange(n_critics)[:, None]
                            critic_idxs = jnp.repeat(critic_idxs, self.nr_atoms, axis=1)  
                            l_idxs = jnp.repeat(l[None, :], n_critics, axis=0)
                            u_idxs = jnp.repeat(u[None, :], n_critics, axis=0)

                            proj_dist = proj_dist.at[(critic_idxs, l_idxs)].add(next_dist * wt_l)
                            proj_dist = proj_dist.at[(critic_idxs, u_idxs)].add(next_dist * wt_u)

                            qf_next_target_value = jnp.sum(proj_dist * q_support, axis=1)  # (2,)

                            if self.clipped_double_q_learning:
                                qf_next_target_dist = jnp.where(qf_next_target_value[0] < qf_next_target_value[1], proj_dist[0], proj_dist[1])  # (nr_atoms,)
                                qf1_next_target_dist = qf_next_target_dist
                                qf2_next_target_dist = qf_next_target_dist
                            else:
                                qf1_next_target_dist = proj_dist[0]
                                qf2_next_target_dist = proj_dist[1]

                            current_q = self.critic.apply(critic_params, normalized_state, action)  # (2, nr_atoms)
                            
                            q1_loss = -jnp.sum(qf1_next_target_dist * jax.nn.log_softmax(current_q[0]), axis=-1)
                            q2_loss = -jnp.sum(qf2_next_target_dist * jax.nn.log_softmax(current_q[1]), axis=-1)
                            loss = q1_loss + q2_loss

                            # Create metrics
                            metrics = {
                                "loss/q_loss": loss,
                                "q/q_max": jnp.max(qf_next_target_value),
                                "q/q_min": jnp.min(qf_next_target_value),
                            }

                            return loss, (metrics)
                        
                        def policy_loss_fn(policy_params, critic_params, normalized_state):
                            # Policy loss
                            action = self.policy.apply(policy_params, normalized_state)

                            q_values = self.critic.apply(critic_params, normalized_state, action)
                            q_values = jnp.sum(jax.nn.softmax(q_values) * jnp.linspace(self.v_min, self.v_max, self.nr_atoms), axis=-1)
                            if self.clipped_double_q_learning:
                                processed_q_value = jnp.min(q_values, axis=0)
                            else:
                                processed_q_value = jnp.mean(q_values, axis=0)

                            loss = -jnp.mean(processed_q_value)

                            # Create metrics
                            metrics = {
                                "loss/policy_loss": loss,
                            }

                            return loss, (metrics)
                        

                        vmap_critic_loss_fn = jax.vmap(critic_loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
                        safe_mean = lambda x: jnp.mean(x) if x is not None else x
                        mean_vmapped_critic_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_critic_loss_fn(*a, **k))
                        grad_critic_loss_fn = jax.value_and_grad(mean_vmapped_critic_loss_fn, argnums=(1,), has_aux=True)

                        vmap_policy_loss_fn = jax.vmap(policy_loss_fn, in_axes=(None, None, 0), out_axes=0)
                        mean_vmapped_policy_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_policy_loss_fn(*a, **k))
                        grad_policy_loss_fn = jax.value_and_grad(mean_vmapped_policy_loss_fn, argnums=(0,), has_aux=True)

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
                        update_keys = jax.random.split(noise_key, self.nr_critic_updates_per_step * self.batch_size)
                        update_keys = update_keys.reshape(self.nr_critic_updates_per_step, self.batch_size, -1)
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
                                keys_for_update = update_keys[update_idx]

                                (loss, (critic_metrics)), (critic_gradients,) = grad_critic_loss_fn(
                                    policy_state.params, critic_state.params, critic_state.target_params,
                                    normalized_states, normalized_next_states, actions, rewards, dones, truncations, effective_n_steps,
                                    keys_for_update)

                                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                                critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))

                                if self.max_grad_norm != -1.0:
                                    lr = critic_state.opt_state[1].hyperparams["learning_rate"]
                                else:
                                    lr = critic_state.opt_state.hyperparams["learning_rate"]
                                critic_metrics["lr/learning_rate"] = lr
                                critic_metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                                update_idx += 1

                            normalized_states_for_policy = normalized_states

                            (loss, (policy_metrics)), (policy_gradients,) = grad_policy_loss_fn(
                                policy_state.params, critic_state.params, normalized_states_for_policy)

                            policy_state = policy_state.apply_gradients(grads=policy_gradients)
                            policy_metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)

                        metrics = {**critic_metrics, **policy_metrics}

                        return (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key), (env_state.info, metrics)
                        
                    key, subkey = jax.random.split(key)
                    learning_iteration_carry, info_and_optimization_metrics = jax.lax.scan(learning_iteration, (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, subkey), jnp.arange(self.nr_updates_per_logging_iteration))
                    policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = learning_iteration_carry
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

                    return (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key), None

                key, subkey = jax.random.split(key)
                logging_iteration_carry, _ = jax.lax.scan(logging_iteration, (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, subkey), jnp.arange(self.nr_loggings_per_eval_save_iteration))
                policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key = logging_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(carry, _):
                        policy_state, eval_env_state = carry
                        if self.enable_observation_normalization:
                            eval_normalized_observation = (eval_env_state.next_observation - observation_normalizer_state["running_mean"]) / (observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)
                        else:
                            eval_normalized_observation = eval_env_state.next_observation
                        eval_action = self.policy.apply(policy_state.params, eval_normalized_observation)
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
                    def save_with_check(policy_state, critic_state, observation_normalizer_state):
                        self.save(policy_state, critic_state, observation_normalizer_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, observation_normalizer_state)

                
                return (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key), None

            jax.lax.scan(eval_save_iteration, (policy_state, critic_state, observation_normalizer_state, replay_buffer, env_state, noise_scales, key), jnp.arange(self.nr_eval_save_iterations))
            

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


    def save(self, policy_state, critic_state, observation_normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
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
        model = FastTD3(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
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
            action = self.policy.apply(self.policy_state.params, normalized_observation)
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
