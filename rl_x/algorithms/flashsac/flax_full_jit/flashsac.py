import os
import shutil
import json
import math
from copy import deepcopy
import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.evaluation import get_evaluation_termination_counts, get_evaluation_termination_metrics
from rl_x.algorithms.flashsac.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.flashsac.flax_full_jit.policy import get_policy
from rl_x.algorithms.flashsac.flax_full_jit.critic import get_critic
from rl_x.algorithms.flashsac.flax_full_jit.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.flashsac.flax_full_jit.layers import project_params
from rl_x.algorithms.flashsac.flax_full_jit.rl_train_state import RLTrainState
from rl_x.algorithms.flashsac.flax_full_jit import reward_normalizer
from rl_x.algorithms.flashsac.flax_full_jit import noise_repeat

rlx_logger = logging.getLogger("rl_x")


class FlashSAC:
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
        self.render_callback_type = getattr(config.environment, 'render_callback_type', 'io_callback')
        self.learning_rate_init = config.algorithm.learning_rate_init
        self.learning_rate_peak = config.algorithm.learning_rate_peak
        self.learning_rate_end = config.algorithm.learning_rate_end
        self.learning_rate_warmup_steps = config.algorithm.learning_rate_warmup_steps
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.policy_delay = config.algorithm.policy_delay
        self.gamma = config.algorithm.gamma
        self.n_steps = config.algorithm.n_steps
        self.tau = config.algorithm.tau
        self.nr_atoms = config.algorithm.nr_atoms
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.normalized_g_max = config.algorithm.normalized_g_max
        self.normalize_reward = config.algorithm.normalize_reward
        self.noise_zeta_mu = config.algorithm.noise_zeta_mu
        self.noise_zeta_max = config.algorithm.noise_zeta_max
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.os_shape = self.train_env.single_observation_space.shape
        self.action_dim = self.train_env.single_action_space.shape[0]
        self.horizon = self.train_env.horizon

        target_sigma = config.algorithm.target_entropy_sigma
        self.target_entropy = 0.5 * self.action_dim * math.log(2.0 * math.pi * math.e * target_sigma ** 2)

        self.capacity = self.buffer_size // self.nr_envs
        if self.total_timesteps % self.nr_envs != 0:
            raise ValueError("The total number of timesteps must be a multiple of the number of environments.")
        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")
        if self.evaluation_and_save_frequency != -1:
            if self.evaluation_and_save_frequency % self.logging_frequency != 0:
                raise ValueError("The evaluation and save frequency must be a multiple of the logging frequency.")
            if self.evaluation_and_save_frequency % self.nr_envs != 0:
                raise ValueError("The evaluation and save frequency must be a multiple of the number of environments.")
        if self.learning_starts > self.total_timesteps:
            raise ValueError("The number of learning-start timesteps cannot exceed the total number of timesteps.")
        if self.capacity < self.n_steps:
            raise ValueError("The replay buffer must hold at least n_steps transitions per environment.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key, reset_key, noise_key = jax.random.split(self.key, 6)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        self.entropy_coefficient = EntropyCoefficient(initial_value=config.algorithm.init_entropy_coefficient)

        total_updates = max(1, (self.total_timesteps // self.nr_envs) * self.updates_per_step)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.learning_rate_init,
            peak_value=self.learning_rate_peak,
            warmup_steps=self.learning_rate_warmup_steps,
            decay_steps=max(total_updates, 1),
            end_value=self.learning_rate_end,
        )

        env_state = self.train_env.reset(reset_key, False)
        dummy_obs = env_state.next_observation
        dummy_action = jnp.zeros(dummy_obs.shape[:-1] + (self.action_dim,), dtype=jnp.float32)

        policy_init = self.policy.init({"params": policy_key, "batch_stats": policy_key}, dummy_obs, train=False)
        critic_init = self.critic.init({"params": critic_key, "batch_stats": critic_key}, dummy_obs, dummy_action, train=False)
        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=project_params(policy_init["params"]),
            batch_stats=policy_init["batch_stats"],
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule),
        )
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=project_params(critic_init["params"]),
            batch_stats=critic_init["batch_stats"],
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule),
        )
        self.target_critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=project_params(critic_init["params"]),
            batch_stats=critic_init["batch_stats"],
            tx=optax.set_to_zero(),
        )
        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule),
        )

        self.zeta_cdf = noise_repeat.build_zeta_cdf(self.noise_zeta_mu, self.noise_zeta_max)
        self.initial_noise_state = noise_repeat.init_noise_state(self.nr_envs, self.action_dim, noise_key)
        self.initial_reward_normalizer_state = reward_normalizer.init_reward_normalizer_state(self.nr_envs)
        self.initial_update_step_count = jnp.zeros((), dtype=jnp.int32)

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
            target_critic_state = self.target_critic_state
            entropy_coefficient_state = self.entropy_coefficient_state
            noise_state = self.initial_noise_state
            reward_norm_state = self.initial_reward_normalizer_state
            update_step_count = self.initial_update_step_count

            replay_buffer = {
                "states": jnp.zeros((self.capacity, self.nr_envs) + self.os_shape, dtype=jnp.float32),
                "next_states": jnp.zeros((self.capacity, self.nr_envs) + self.os_shape, dtype=jnp.float32),
                "actions": jnp.zeros((self.capacity, self.nr_envs, self.action_dim), dtype=jnp.float32),
                "rewards": jnp.zeros((self.capacity, self.nr_envs), dtype=jnp.float32),
                "dones": jnp.zeros((self.capacity, self.nr_envs), dtype=jnp.float32),
                "truncations": jnp.zeros((self.capacity, self.nr_envs), dtype=jnp.float32),
                "pos": jnp.zeros((), dtype=jnp.int32),
                "size": jnp.zeros((), dtype=jnp.int32),
            }

            def collect_step(policy_state, env_state, noise_state, reward_norm_state, replay_buffer, key, use_random_action):
                key, action_key, noise_key = jax.random.split(key, 3)
                observation = env_state.next_observation

                def get_random_action(noise_state):
                    action = jax.random.uniform(action_key, (self.nr_envs, self.action_dim), minval=-1.0, maxval=1.0)
                    return action, noise_state

                def get_policy_action(noise_state):
                    mean, std = self.policy.apply(
                        {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                        observation, train=False,
                    )
                    noise_state = noise_repeat.step_noise(noise_state, noise_key, self.zeta_cdf)
                    action = self.policy.apply(
                        {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                        mean, std, noise_state["noise"], 1.0,
                        method=self.policy.sample_with_noise,
                    )
                    return action, noise_state

                action, noise_state = jax.lax.cond(
                    use_random_action, get_random_action, get_policy_action, noise_state,
                )
                env_state = self.train_env.step(env_state, self.get_processed_action(action))
                done = (env_state.terminated | env_state.truncated).astype(jnp.float32)

                if self.normalize_reward:
                    reward_norm_state = reward_normalizer.update_reward_normalizer(
                        reward_norm_state, env_state.reward, env_state.terminated, env_state.truncated, self.gamma,
                    )

                replay_buffer["states"] = replay_buffer["states"].at[replay_buffer["pos"]].set(observation)
                replay_buffer["next_states"] = replay_buffer["next_states"].at[replay_buffer["pos"]].set(env_state.actual_next_observation)
                replay_buffer["actions"] = replay_buffer["actions"].at[replay_buffer["pos"]].set(action)
                replay_buffer["rewards"] = replay_buffer["rewards"].at[replay_buffer["pos"]].set(env_state.reward)
                replay_buffer["dones"] = replay_buffer["dones"].at[replay_buffer["pos"]].set(done)
                replay_buffer["truncations"] = replay_buffer["truncations"].at[replay_buffer["pos"]].set(env_state.truncated.astype(jnp.float32))
                replay_buffer["pos"] = (replay_buffer["pos"] + 1) % self.capacity
                replay_buffer["size"] = jnp.minimum(replay_buffer["size"] + 1, self.capacity)

                if self.render:
                    if self.render_callback_type == "debug_callback":
                        jax.debug.callback(self.train_env.render, env_state)
                    else:
                        def render(env_state):
                            return self.train_env.render(env_state)
                        env_state = jax.experimental.io_callback(render, env_state, env_state)

                return env_state, noise_state, reward_norm_state, replay_buffer, key

            def critic_loss_fn(critic_params, batch_stats, target_critic_params, target_critic_batch_stats, policy_params, policy_batch_stats, entropy_coefficient_params, states, next_states, actions, rewards, dones, truncations, effective_n_steps, key):
                next_mean, next_std = self.policy.apply(
                    {"params": policy_params, "batch_stats": policy_batch_stats},
                    next_states, train=False,
                )
                next_action, next_log_prob = self.policy.apply(
                    {"params": policy_params, "batch_stats": policy_batch_stats},
                    next_mean, next_std, key,
                    method=self.policy.sample_and_log_prob,
                )
                alpha = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_params))

                all_observations = jnp.concatenate([states, next_states], axis=0)
                all_actions = jnp.concatenate([actions, next_action], axis=0)

                (_, target_log_probabilities), target_critic_state_update = self.critic.apply(
                    {"params": target_critic_params, "batch_stats": target_critic_batch_stats},
                    all_observations, all_actions, train=True,
                    mutable=["batch_stats"],
                )
                next_log_probabilities = target_log_probabilities[:, states.shape[0]:, :]
                next_values = jnp.sum(
                    jnp.exp(next_log_probabilities) * jnp.linspace(self.v_min, self.v_max, self.nr_atoms, dtype=jnp.float32), axis=-1,
                )
                minimum_value_indices = jnp.argmin(next_values, axis=0)
                selected_next_log_probabilities = jnp.take_along_axis(
                    next_log_probabilities, minimum_value_indices[None, :, None], axis=0,
                )[0]

                discount = (self.gamma ** effective_n_steps) * (1.0 - dones * (1.0 - truncations))
                bin_values = jnp.linspace(self.v_min, self.v_max, self.nr_atoms, dtype=jnp.float32)
                target_bin_values = rewards[:, None] + discount[:, None] * (bin_values[None, :] - (alpha * next_log_prob)[:, None])
                target_bin_values = jnp.clip(target_bin_values, self.v_min, self.v_max)
                bin_width = (self.v_max - self.v_min) / (self.nr_atoms - 1)
                target_bin_indices = (target_bin_values - self.v_min) / bin_width
                lower_bin_indices = jnp.floor(target_bin_indices).astype(jnp.int32)
                upper_bin_indices = jnp.clip(lower_bin_indices + 1, 0, self.nr_atoms - 1)
                upper_bin_weights = target_bin_indices - lower_bin_indices.astype(jnp.float32)
                next_probabilities = jnp.exp(selected_next_log_probabilities)
                lower_bin_weights = next_probabilities * (1.0 - upper_bin_weights)
                upper_bin_weights = next_probabilities * upper_bin_weights
                batch_indices = jnp.broadcast_to(jnp.arange(rewards.shape[0])[:, None], lower_bin_indices.shape)
                target_probabilities = jnp.zeros((rewards.shape[0], self.nr_atoms), dtype=jnp.float32)
                target_probabilities = target_probabilities.at[(batch_indices, lower_bin_indices)].add(lower_bin_weights)
                target_probabilities = target_probabilities.at[(batch_indices, upper_bin_indices)].add(upper_bin_weights)

                (_, predicted_log_probabilities), critic_state_update = self.critic.apply(
                    {"params": critic_params, "batch_stats": batch_stats},
                    all_observations, all_actions, train=True,
                    mutable=["batch_stats"],
                )
                predicted_log_probabilities = predicted_log_probabilities[:, :states.shape[0], :]
                cross_entropy = -jnp.sum(target_probabilities[None, :, :] * predicted_log_probabilities, axis=-1)
                loss = jnp.mean(cross_entropy)
                return loss, (critic_state_update, target_critic_state_update, jnp.mean(next_values))

            def policy_loss_fn(policy_params, batch_stats, critic_params, critic_batch_stats, entropy_coefficient_params, states, next_states, key):
                all_observations = jnp.concatenate([states, next_states], axis=0)
                (all_means, all_stds), policy_state_update = self.policy.apply(
                    {"params": policy_params, "batch_stats": batch_stats},
                    all_observations, train=True,
                    mutable=["batch_stats"],
                )
                mean = all_means[:states.shape[0]]
                std = all_stds[:states.shape[0]]
                action, log_prob = self.policy.apply(
                    {"params": policy_params, "batch_stats": batch_stats},
                    mean, std, key,
                    method=self.policy.sample_and_log_prob,
                )

                (q_values, _), _ = self.critic.apply(
                    {"params": critic_params, "batch_stats": critic_batch_stats},
                    states, action, train=False,
                    mutable=["batch_stats"],
                )
                q = jnp.minimum(q_values[0], q_values[1])
                alpha = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_params))
                loss = jnp.mean(alpha * log_prob - q)
                entropy = -jnp.mean(log_prob)
                return loss, (policy_state_update, entropy, jnp.mean(q))

            def entropy_loss_fn(entropy_coefficient_params, entropy):
                alpha = self.entropy_coefficient.apply(entropy_coefficient_params)
                loss = alpha * (entropy - self.target_entropy)
                return loss, alpha

            critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)
            policy_grad_fn = jax.value_and_grad(policy_loss_fn, argnums=0, has_aux=True)
            entropy_grad_fn = jax.value_and_grad(entropy_loss_fn, argnums=0, has_aux=True)

            def sample_batch(replay_buffer, reward_norm_state, key):
                idx_key_t, idx_key_e = jax.random.split(key)
                if self.n_steps == 1:
                    idx1 = jax.random.randint(idx_key_t, (self.batch_size,), 0, replay_buffer["size"])
                    idx2 = jax.random.randint(idx_key_e, (self.batch_size,), 0, self.nr_envs)
                    rewards = replay_buffer["rewards"][idx1, idx2]
                    if self.normalize_reward:
                        rewards = reward_normalizer.normalize_reward(reward_norm_state, rewards, self.normalized_g_max)
                    return (
                        replay_buffer["states"][idx1, idx2],
                        replay_buffer["next_states"][idx1, idx2],
                        replay_buffer["actions"][idx1, idx2],
                        rewards,
                        replay_buffer["dones"][idx1, idx2],
                        replay_buffer["truncations"][idx1, idx2],
                        jnp.ones((self.batch_size,), dtype=jnp.float32),
                    )

                max_start = jnp.where(
                    replay_buffer["size"] >= self.capacity,
                    self.capacity,
                    jnp.maximum(1, replay_buffer["size"] - self.n_steps + 1),
                )
                last_idx = (replay_buffer["pos"] - 1) % self.capacity

                idx1 = jax.random.randint(idx_key_t, (self.batch_size,), 0, max_start)
                idx2 = jax.random.randint(idx_key_e, (self.batch_size,), 0, self.nr_envs)
                steps = jnp.arange(self.n_steps)
                all_indices = (idx1[:, None] + steps) % self.capacity
                env_indices = jnp.broadcast_to(idx2[:, None], all_indices.shape)
                flat_t = all_indices.reshape(-1)
                flat_e = env_indices.reshape(-1)
                all_rewards = replay_buffer["rewards"][flat_t, flat_e].reshape(all_indices.shape)
                all_dones = replay_buffer["dones"][flat_t, flat_e].reshape(all_indices.shape)
                all_truncations = replay_buffer["truncations"][flat_t, flat_e].reshape(all_indices.shape)
                all_truncations = jnp.where(
                    (replay_buffer["size"] >= self.capacity) & (all_indices == last_idx) & (all_dones <= 0.0),
                    jnp.ones_like(all_truncations),
                    all_truncations,
                )

                all_episode_ends = jnp.maximum(all_dones, all_truncations)
                zeros_first = jnp.zeros((all_episode_ends.shape[0], 1), dtype=all_episode_ends.dtype)
                all_dones_shifted = jnp.concatenate([zeros_first, all_episode_ends[:, :-1]], axis=-1)
                done_masks = jnp.cumprod(1 - all_dones_shifted, axis=-1)
                effective_n_steps = jnp.sum(done_masks, axis=-1)
                discounts = self.gamma ** jnp.arange(self.n_steps)
                rewards = jnp.sum(all_rewards * done_masks * discounts, axis=-1)

                all_dones_int = (all_dones > 0.0).astype(jnp.int32)
                all_trunc_int = (all_truncations > 0.0).astype(jnp.int32)
                first_done = jnp.where(jnp.sum(all_dones_int, axis=-1) == 0, self.n_steps - 1, jnp.argmax(all_dones_int, axis=-1))
                first_trunc = jnp.where(jnp.sum(all_trunc_int, axis=-1) == 0, self.n_steps - 1, jnp.argmax(all_trunc_int, axis=-1))
                final_offset = jnp.minimum(first_done, first_trunc)
                final_time_indices = jnp.take_along_axis(all_indices, final_offset[:, None], axis=-1).squeeze(-1)
                next_states = replay_buffer["next_states"][final_time_indices, idx2]
                dones = replay_buffer["dones"][final_time_indices, idx2]
                truncations = replay_buffer["truncations"][final_time_indices, idx2]
                truncations = jnp.where(
                    (replay_buffer["size"] >= self.capacity) & (final_time_indices == last_idx) & (dones <= 0.0),
                    jnp.ones_like(truncations),
                    truncations,
                )

                states = replay_buffer["states"][idx1, idx2]
                actions = replay_buffer["actions"][idx1, idx2]
                if self.normalize_reward:
                    rewards = reward_normalizer.normalize_reward(reward_norm_state, rewards, self.normalized_g_max)
                return states, next_states, actions, rewards, dones, truncations, effective_n_steps

            def update_step(carry, _):
                policy_state, critic_state, target_critic_state, entropy_coefficient_state, replay_buffer, reward_norm_state, update_step_count, key = carry
                key, sample_key, policy_key, critic_key = jax.random.split(key, 4)
                states, next_states, actions, rewards, dones, truncations, effective_n_steps = sample_batch(
                    replay_buffer, reward_norm_state, sample_key,
                )

                def apply_policy_update(carry):
                    policy_state, entropy_coefficient_state = carry
                    (policy_loss, (policy_state_update, entropy, policy_q_mean)), policy_gradients = policy_grad_fn(
                        policy_state.params, policy_state.batch_stats,
                        critic_state.params, critic_state.batch_stats,
                        entropy_coefficient_state.params,
                        states, next_states, policy_key,
                    )
                    policy_state = policy_state.apply_gradients(grads=policy_gradients)
                    policy_state = policy_state.replace(
                        params=project_params(policy_state.params),
                        batch_stats=policy_state_update["batch_stats"],
                    )
                    (entropy_loss, alpha), entropy_gradients = entropy_grad_fn(
                        entropy_coefficient_state.params, stop_gradient(entropy),
                    )
                    entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)
                    metrics = {
                        "loss/policy_loss": policy_loss,
                        "loss/entropy_loss": entropy_loss,
                        "entropy/entropy": entropy,
                        "entropy/alpha": alpha,
                        "q_value/policy_q_mean": policy_q_mean,
                    }
                    return policy_state, entropy_coefficient_state, metrics

                def skip_policy_update(carry):
                    policy_state, entropy_coefficient_state = carry
                    metrics = {
                        "loss/policy_loss": jnp.nan,
                        "loss/entropy_loss": jnp.nan,
                        "entropy/entropy": jnp.nan,
                        "entropy/alpha": jnp.nan,
                        "q_value/policy_q_mean": jnp.nan,
                    }
                    return policy_state, entropy_coefficient_state, metrics

                policy_state, entropy_coefficient_state, policy_metrics = jax.lax.cond(
                    update_step_count % self.policy_delay == 0,
                    apply_policy_update,
                    skip_policy_update,
                    (policy_state, entropy_coefficient_state),
                )

                (critic_loss, (critic_state_update, target_critic_state_update, target_q_mean)), critic_gradients = critic_grad_fn(
                    critic_state.params, critic_state.batch_stats,
                    target_critic_state.params, target_critic_state.batch_stats,
                    policy_state.params, policy_state.batch_stats,
                    entropy_coefficient_state.params,
                    states, next_states, actions, rewards, dones, truncations, effective_n_steps, critic_key,
                )
                critic_state = critic_state.apply_gradients(grads=critic_gradients)
                critic_state = critic_state.replace(
                    params=project_params(critic_state.params),
                    batch_stats=critic_state_update["batch_stats"],
                )
                new_target_params = jax.tree_util.tree_map(
                    lambda parameter, target_parameter: self.tau * parameter + (1.0 - self.tau) * target_parameter,
                    critic_state.params, target_critic_state.params,
                )
                target_critic_state = target_critic_state.replace(
                    params=new_target_params,
                    batch_stats=target_critic_state_update["batch_stats"],
                )

                metrics = {
                    **policy_metrics,
                    "loss/critic_loss": critic_loss,
                    "q_value/target_q_mean": target_q_mean,
                    "lr/policy_learning_rate": policy_state.opt_state.hyperparams["learning_rate"],
                    "lr/critic_learning_rate": critic_state.opt_state.hyperparams["learning_rate"],
                }
                return (
                    policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                    replay_buffer, reward_norm_state, update_step_count + 1, key,
                ), metrics

            def rollout_step(carry, _):
                policy_state, critic_state, target_critic_state, entropy_coefficient_state, env_state, noise_state, reward_norm_state, replay_buffer, update_step_count, rollout_count, key = carry
                global_step_before = rollout_count * self.nr_envs
                env_state, noise_state, reward_norm_state, replay_buffer, key = collect_step(
                    policy_state, env_state, noise_state, reward_norm_state, replay_buffer, key,
                    global_step_before < self.learning_starts,
                )
                rollout_count += 1

                def apply_updates(carry):
                    policy_state, critic_state, target_critic_state, entropy_coefficient_state, replay_buffer, reward_norm_state, update_step_count, key = carry
                    carry, update_metrics = jax.lax.scan(
                        update_step,
                        carry,
                        jnp.arange(self.updates_per_step),
                    )
                    update_metrics = jax.tree.map(lambda value: jnp.nanmean(value), update_metrics)
                    return carry, update_metrics

                def skip_updates(carry):
                    metrics = {
                        "loss/policy_loss": jnp.nan,
                        "loss/entropy_loss": jnp.nan,
                        "loss/critic_loss": jnp.nan,
                        "entropy/entropy": jnp.nan,
                        "entropy/alpha": jnp.nan,
                        "q_value/policy_q_mean": jnp.nan,
                        "q_value/target_q_mean": jnp.nan,
                        "lr/policy_learning_rate": jnp.nan,
                        "lr/critic_learning_rate": jnp.nan,
                    }
                    return carry, metrics

                update_carry, update_metrics = jax.lax.cond(
                    (rollout_count * self.nr_envs >= self.learning_starts) & (replay_buffer["size"] >= self.n_steps),
                    apply_updates,
                    skip_updates,
                    (
                        policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                        replay_buffer, reward_norm_state, update_step_count, key,
                    ),
                )
                policy_state, critic_state, target_critic_state, entropy_coefficient_state, replay_buffer, reward_norm_state, update_step_count, key = update_carry
                return (
                    policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                    env_state, noise_state, reward_norm_state, replay_buffer,
                    update_step_count, rollout_count, key,
                ), (env_state.info, update_metrics)

            def evaluate_and_save(carry, global_step):
                policy_state, critic_state, target_critic_state, entropy_coefficient_state, env_state, noise_state, reward_norm_state, replay_buffer, update_step_count, rollout_count, key = carry
                if self.evaluation_and_save_frequency != -1:
                    should_evaluate_and_save = global_step % self.evaluation_and_save_frequency == 0

                    if self.evaluation_active:
                        def evaluate(key):
                            key, reset_key = jax.random.split(key)
                            reset_keys = jax.random.split(reset_key, self.nr_envs)
                            eval_env_state = self.eval_env.reset(reset_keys, True)
                            eval_termination_counts = jax.tree.map(jnp.zeros_like, get_evaluation_termination_counts(eval_env_state))

                            def single_eval_rollout(eval_carry, _):
                                policy_state, eval_env_state, eval_termination_counts = eval_carry
                                eval_mean, _ = self.policy.apply(
                                    {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                                    eval_env_state.next_observation, train=False,
                                )
                                eval_action = self.policy.deterministic_action(eval_mean)
                                eval_env_state = self.eval_env.step(eval_env_state, self.get_processed_action(eval_action))
                                eval_termination_counts = jax.tree.map(jnp.add, eval_termination_counts, get_evaluation_termination_counts(eval_env_state))
                                return (policy_state, eval_env_state, eval_termination_counts), None

                            (_, eval_env_state, eval_termination_counts), _ = jax.lax.scan(
                                single_eval_rollout,
                                (policy_state, eval_env_state, eval_termination_counts),
                                jnp.arange(self.horizon),
                            )
                            eval_metrics = {
                                "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                                "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                            }
                            eval_metrics.update(get_evaluation_termination_metrics(eval_termination_counts))

                            def eval_callback(args):
                                metrics, step = args
                                step = int(step)
                                self.start_logging(step)
                                for name, value in metrics.items():
                                    self.log(name, np.asarray(value), step)
                                self.end_logging()

                            jax.debug.callback(eval_callback, (eval_metrics, global_step))
                            return key

                        key = jax.lax.cond(should_evaluate_and_save, evaluate, lambda key: key, key)

                    if self.save_model:
                        def save_model(operand):
                            policy_state, critic_state, target_critic_state, entropy_coefficient_state, reward_norm_state, update_step_count = operand

                            def save_with_check(policy_state, critic_state, target_critic_state, entropy_coefficient_state, reward_norm_state, update_step_count):
                                self.save(
                                    policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                                    reward_norm_state, update_step_count,
                                )

                            jax.debug.callback(
                                save_with_check,
                                policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                                reward_norm_state, update_step_count,
                            )
                            return operand

                        jax.lax.cond(
                            should_evaluate_and_save,
                            save_model,
                            lambda operand: operand,
                            (policy_state, critic_state, target_critic_state, entropy_coefficient_state, reward_norm_state, update_step_count),
                        )

                return (
                    policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                    env_state, noise_state, reward_norm_state, replay_buffer,
                    update_step_count, rollout_count, key,
                )

            rollouts_per_logging = self.logging_frequency // self.nr_envs
            total_rollouts = self.total_timesteps // self.nr_envs
            nr_logging_iterations = total_rollouts // rollouts_per_logging
            remaining_rollouts = total_rollouts % rollouts_per_logging

            def logging_iteration(carry, _):
                carry, (infos, metrics) = jax.lax.scan(
                    rollout_step,
                    carry,
                    jnp.arange(rollouts_per_logging),
                )
                combined_metrics = {
                    **jax.tree.map(lambda value: jnp.mean(value), infos),
                    **jax.tree.map(lambda value: jnp.nanmean(value), metrics),
                }
                update_step_count = carry[8]
                global_step = carry[9] * self.nr_envs

                def callback(args):
                    metrics, update_count, step, parallel_seed_id = args
                    current_time = time.time()
                    metrics["time/sps"] = int(self.logging_frequency / (current_time - self.last_time[parallel_seed_id]))
                    self.last_time[parallel_seed_id] = current_time
                    step = int(step)
                    metrics["steps/nr_env_steps"] = step
                    metrics["steps/nr_updates"] = int(update_count)
                    self.start_logging(step)
                    for name, value in metrics.items():
                        value = np.asarray(value)
                        if not np.issubdtype(value.dtype, np.floating) or not np.isnan(value).all():
                            self.log(name, value, step)
                    self.end_logging()

                jax.debug.callback(callback, (combined_metrics, update_step_count, global_step, parallel_seed_id))
                carry = evaluate_and_save(carry, global_step)
                return carry, None

            carry = (
                policy_state, critic_state, target_critic_state, entropy_coefficient_state,
                env_state, noise_state, reward_norm_state, replay_buffer,
                update_step_count, jnp.zeros((), dtype=jnp.int32), key,
            )
            carry, _ = jax.lax.scan(logging_iteration, carry, jnp.arange(nr_logging_iterations))

            if remaining_rollouts > 0:
                carry, (infos, metrics) = jax.lax.scan(
                    rollout_step,
                    carry,
                    jnp.arange(remaining_rollouts),
                )
                combined_metrics = {
                    **jax.tree.map(lambda value: jnp.mean(value), infos),
                    **jax.tree.map(lambda value: jnp.nanmean(value), metrics),
                }
                update_step_count = carry[8]
                global_step = carry[9] * self.nr_envs

                def final_callback(args):
                    metrics, update_count, step, parallel_seed_id = args
                    current_time = time.time()
                    metrics["time/sps"] = int((remaining_rollouts * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                    self.last_time[parallel_seed_id] = current_time
                    step = int(step)
                    metrics["steps/nr_env_steps"] = step
                    metrics["steps/nr_updates"] = int(update_count)
                    self.start_logging(step)
                    for name, value in metrics.items():
                        value = np.asarray(value)
                        if not np.issubdtype(value.dtype, np.floating) or not np.isnan(value).all():
                            self.log(name, value, step)
                    self.end_logging()

                jax.debug.callback(final_callback, (combined_metrics, update_step_count, global_step, parallel_seed_id))
                carry = evaluate_and_save(carry, global_step)

            return carry[0], carry[1], carry[2], carry[3], carry[6], carry[8], carry[10]

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        result = jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        self.policy_state = jax.tree.map(lambda value: value[0], result[0])
        self.critic_state = jax.tree.map(lambda value: value[0], result[1])
        self.target_critic_state = jax.tree.map(lambda value: value[0], result[2])
        self.entropy_coefficient_state = jax.tree.map(lambda value: value[0], result[3])
        self.initial_reward_normalizer_state = jax.tree.map(lambda value: value[0], result[4])
        self.initial_update_step_count = result[5][0]
        self.key = result[6][0]
        rlx_logger.info(f"Average time: {max([time.time() - t for t in self.start_time]):.2f} s")


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


    def save(self, policy_state, critic_state, target_critic_state, entropy_coefficient_state, reward_normalizer_state, update_step_count):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "target_critic": target_critic_state,
            "entropy_coefficient": entropy_coefficient_state,
            "reward_normalizer": reward_normalizer_state,
            "update_step_count": np.asarray(update_step_count, dtype=np.int64),
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
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FlashSAC(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "target_critic": model.target_critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
            "reward_normalizer": model.initial_reward_normalizer_state,
            "update_step_count": np.asarray(0, dtype=np.int64),
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.target_critic_state = checkpoint["target_critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.initial_reward_normalizer_state = checkpoint["reward_normalizer"]
        model.initial_update_step_count = jnp.asarray(checkpoint["update_step_count"], dtype=jnp.int32)
        shutil.rmtree(checkpoint_dir)
        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state):
            mean, _ = self.policy.apply(
                {"params": self.policy_state.params, "batch_stats": self.policy_state.batch_stats},
                env_state.next_observation, train=False,
            )
            action = self.policy.apply(
                {"params": self.policy_state.params, "batch_stats": self.policy_state.batch_stats},
                mean, method=self.policy.deterministic_action,
            )
            env_state = self.train_env.step(env_state, self.get_processed_action(action))
            return env_state

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, True)
        while True:
            env_state = rollout(env_state)
            if self.render:
                env_state = self.train_env.render(env_state)


    def general_properties():
        return GeneralProperties
