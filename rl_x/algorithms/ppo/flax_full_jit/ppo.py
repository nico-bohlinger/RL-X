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

from rl_x.algorithms.ppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.ppo.flax_full_jit.policy import get_policy
from rl_x.algorithms.ppo.flax_full_jit.critic import Critic, VectorCritic, get_critic
from rl_x.algorithms.fastmpo.flax_full_jit.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class PPO:
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
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.mpo_aux_loss_coefficient = getattr(config.algorithm, "mpo_aux_loss_coefficient", 0.0)
        self.mpo_aux_action_samples = getattr(config.algorithm, "mpo_aux_action_samples", 8)
        self.mpo_aux_temperature = getattr(config.algorithm, "mpo_aux_temperature", 1.0)
        self.mpo_aux_q_critic_coef = getattr(config.algorithm, "mpo_aux_q_critic_coef", 1.0)
        self.mpo_aux_start_step = getattr(config.algorithm, "mpo_aux_start_step", 0)
        self.mpo_aux_end_step = getattr(config.algorithm, "mpo_aux_end_step", 0)
        self.use_mpo_aux = self.mpo_aux_loss_coefficient > 0.0
        self.mpo_aux_replay_buffer_size_per_env = getattr(config.algorithm, "mpo_aux_replay_buffer_size_per_env", 0)
        self.mpo_aux_replay_batch_size = getattr(config.algorithm, "mpo_aux_replay_batch_size", 8192)
        self.mpo_aux_replay_critic_tau = getattr(config.algorithm, "mpo_aux_replay_critic_tau", 0.005)
        self.use_mpo_aux_replay = self.use_mpo_aux and self.mpo_aux_replay_buffer_size_per_env > 0
        self.nr_value_samples = getattr(config.algorithm, "nr_value_samples", 0)
        self.nr_q_critics = getattr(config.algorithm, "nr_q_critics", 1)
        self.critic_tau = getattr(config.algorithm, "critic_tau", 0.0)
        self.q_critic_reduction = getattr(config.algorithm, "q_critic_reduction", "min")
        self.use_target_q_critic = self.nr_q_critics > 1 and self.critic_tau > 0.0
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.horizon = self.train_env.horizon

        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")
        
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log mutliple wandb runs at the same time.")

        if self.q_critic_reduction not in ["min", "mean"]:
            raise ValueError("Q critic reduction must be min or mean.")

        if self.use_mpo_aux and self.nr_value_samples > 0:
            raise ValueError("The MPO auxiliary requires PPO's ordinary V critic.")

        if self.use_mpo_aux and self.mpo_aux_action_samples < 2:
            raise ValueError("The MPO auxiliary requires at least two action samples.")

        if self.use_mpo_aux and self.mpo_aux_temperature <= 0.0:
            raise ValueError("The MPO auxiliary temperature must be positive.")

        if self.use_mpo_aux and 0 < self.mpo_aux_end_step <= self.mpo_aux_start_step:
            raise ValueError("The MPO auxiliary end step must be greater than its start step.")

        if self.use_mpo_aux_replay and self.mpo_aux_replay_buffer_size_per_env < self.nr_steps:
            raise ValueError("The MPO auxiliary replay buffer must hold at least one PPO rollout.")

        if self.use_mpo_aux_replay and self.mpo_aux_replay_batch_size < 1:
            raise ValueError("The MPO auxiliary replay batch size must be positive.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        if self.use_mpo_aux:
            self.key, policy_key, critic_key, mpo_aux_q_critic_key, reset_key = jax.random.split(self.key, 5)
        else:
            self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, env_state.next_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        critic_init_arguments = (env_state.next_observation,)
        if self.nr_value_samples > 0:
            critic_init_arguments += (jnp.zeros(env_state.next_observation.shape[:-1] + self.as_shape),)
        critic_params = self.critic.init(critic_key, *critic_init_arguments)
        critic_state_arguments = {
            "apply_fn": self.critic.apply,
            "params": critic_params,
            "tx": optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        }
        if self.use_target_q_critic:
            self.critic_state = RLTrainState.create(target_params=critic_params, **critic_state_arguments)
        else:
            self.critic_state = TrainState.create(**critic_state_arguments)

        if self.use_mpo_aux:
            critic_observation_indices = getattr(self.train_env, "critic_observation_indices", jnp.arange(self.train_env.single_observation_space.shape[0]))
            if self.use_mpo_aux_replay:
                self.mpo_aux_q_critic = VectorCritic(2, critic_observation_indices)
            else:
                self.mpo_aux_q_critic = Critic(critic_observation_indices, True)
            mpo_aux_q_critic_params = self.mpo_aux_q_critic.init(
                mpo_aux_q_critic_key,
                env_state.next_observation,
                jnp.zeros(env_state.next_observation.shape[:-1] + self.as_shape),
            )
            mpo_aux_q_critic_state_arguments = {
                "apply_fn": self.mpo_aux_q_critic.apply,
                "params": mpo_aux_q_critic_params,
                "tx": optax.chain(
                    optax.clip_by_global_norm(self.max_grad_norm),
                    optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
                )
            }
            if self.use_mpo_aux_replay:
                self.mpo_aux_q_critic_state = RLTrainState.create(target_params=mpo_aux_q_critic_params, **mpo_aux_q_critic_state_arguments)
            else:
                self.mpo_aux_q_critic_state = TrainState.create(**mpo_aux_q_critic_state_arguments)

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
            if self.use_mpo_aux:
                mpo_aux_q_critic_state = self.mpo_aux_q_critic_state
                mpo_aux_replay_capacity = max(1, self.mpo_aux_replay_buffer_size_per_env)
                mpo_aux_replay_buffer = {
                    "states": jnp.zeros((mpo_aux_replay_capacity, self.nr_envs) + self.os_shape, dtype=jnp.float32),
                    "next_states": jnp.zeros((mpo_aux_replay_capacity, self.nr_envs) + self.os_shape, dtype=jnp.float32),
                    "actions": jnp.zeros((mpo_aux_replay_capacity, self.nr_envs) + self.as_shape, dtype=jnp.float32),
                    "rewards": jnp.zeros((mpo_aux_replay_capacity, self.nr_envs), dtype=jnp.float32),
                    "terminations": jnp.zeros((mpo_aux_replay_capacity, self.nr_envs), dtype=jnp.float32),
                    "pos": jnp.zeros((), dtype=jnp.int32),
                    "size": jnp.zeros((), dtype=jnp.int32),
                }

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                if self.use_mpo_aux:
                    policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key = multi_learning_and_eval_save_iteration_carry
                else:
                    policy_state, critic_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    if self.use_mpo_aux:
                        policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key = learning_iteration_carry
                    else:
                        policy_state, critic_state, env_state, key = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, env_state, key = single_rollout_carry

                        key, action_key = jax.random.split(key)
                        observation = env_state.next_observation
                        action_mean, action_logstd = self.policy.apply(policy_state.params, observation)
                        action_std = jnp.exp(action_logstd)
                        action = action_mean + action_std * jax.random.normal(action_key, shape=action_mean.shape)
                        log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(1)
                        processed_action = self.get_processed_action(action)
                        if self.nr_value_samples > 0:
                            critic_value_params = critic_state.target_params if self.use_target_q_critic else critic_state.params
                            key, value_key = jax.random.split(key)
                            value_actions = action_mean[None, :] + action_std[None, :] * jax.random.normal(value_key, shape=(self.nr_value_samples,) + action_mean.shape)
                            value_q = self.critic.apply(critic_value_params, jnp.repeat(observation[None, :], self.nr_value_samples, axis=0), value_actions).squeeze(-1)
                            if self.nr_q_critics > 1:
                                twin_q_disagreement = jnp.mean(jnp.max(value_q, axis=0) - jnp.min(value_q, axis=0), axis=0)
                                value_q = jnp.min(value_q, axis=0) if self.q_critic_reduction == "min" else jnp.mean(value_q, axis=0)
                            else:
                                twin_q_disagreement = jnp.zeros(value_q.shape[-1])
                            value = jnp.mean(value_q, axis=0)
                            action_q = self.critic.apply(critic_value_params, observation, action).squeeze(-1)
                            if self.nr_q_critics > 1:
                                action_q = jnp.min(action_q, axis=0) if self.q_critic_reduction == "min" else jnp.mean(action_q, axis=0)
                            sampled_action_q_range = jnp.max(value_q, axis=0) - jnp.min(value_q, axis=0)
                            executed_action_advantage = action_q - value
                        else:
                            value = self.critic.apply(critic_state.params, observation).squeeze(-1)
                            sampled_action_q_range = jnp.zeros_like(value)
                            executed_action_advantage = jnp.zeros_like(value)
                            twin_q_disagreement = jnp.zeros_like(value)

                        env_state = self.train_env.step(env_state, processed_action)
                        if self.nr_value_samples > 0:
                            next_action_mean, next_action_logstd = self.policy.apply(policy_state.params, env_state.actual_next_observation)
                            next_action_std = jnp.exp(next_action_logstd)
                            key, next_value_key = jax.random.split(key)
                            next_value_actions = next_action_mean[None, :] + next_action_std[None, :] * jax.random.normal(next_value_key, shape=(self.nr_value_samples,) + next_action_mean.shape)
                            next_value_q = self.critic.apply(critic_value_params, jnp.repeat(env_state.actual_next_observation[None, :], self.nr_value_samples, axis=0), next_value_actions).squeeze(-1)
                            if self.nr_q_critics > 1:
                                next_value_q = jnp.min(next_value_q, axis=0) if self.q_critic_reduction == "min" else jnp.mean(next_value_q, axis=0)
                            next_value = next_value_q.mean(axis=0)
                        else:
                            next_value = jnp.zeros_like(value)
                        transition = (observation, env_state.actual_next_observation, action, env_state.reward, value, next_value, env_state.terminated, log_prob, sampled_action_q_range, executed_action_advantage, twin_q_disagreement, env_state.info)

                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                def render(env_state):
                                    return self.train_env.render(env_state)
                                env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, (policy_state, critic_state, env_state, key), None, self.nr_steps)
                    policy_state, critic_state, env_state, key = single_rollout_carry
                    states, next_states, actions, rewards, values, next_values, terminations, log_probs, sampled_action_q_ranges, executed_action_advantages, twin_q_disagreements, infos = batch

                    if self.use_mpo_aux_replay:
                        replay_indices = (mpo_aux_replay_buffer["pos"] + jnp.arange(self.nr_steps)) % self.mpo_aux_replay_buffer_size_per_env
                        mpo_aux_replay_buffer["states"] = mpo_aux_replay_buffer["states"].at[replay_indices].set(states)
                        mpo_aux_replay_buffer["next_states"] = mpo_aux_replay_buffer["next_states"].at[replay_indices].set(next_states)
                        mpo_aux_replay_buffer["actions"] = mpo_aux_replay_buffer["actions"].at[replay_indices].set(actions)
                        mpo_aux_replay_buffer["rewards"] = mpo_aux_replay_buffer["rewards"].at[replay_indices].set(rewards)
                        mpo_aux_replay_buffer["terminations"] = mpo_aux_replay_buffer["terminations"].at[replay_indices].set(terminations)
                        mpo_aux_replay_buffer["pos"] = (mpo_aux_replay_buffer["pos"] + self.nr_steps) % self.mpo_aux_replay_buffer_size_per_env
                        mpo_aux_replay_buffer["size"] = jnp.minimum(mpo_aux_replay_buffer["size"] + self.nr_steps, self.mpo_aux_replay_buffer_size_per_env)


                    # Calculating advantages and returns
                    if self.nr_value_samples == 0:
                        next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

                    def calculate_gae_advantages(next_values, rewards, values, terminations):
                        def compute_advantages(carry, t):
                            prev_advantage = carry[0]
                            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                            return (advantage,), advantage

                        delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
                        init_advantages = delta[-1]
                        _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1), unroll=True)
                        advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
                        returns = advantages + values
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(next_values, rewards, values, terminations)


                    # Optimizing
                    behavior_policy_params = tree.map_structure(jax.lax.stop_gradient, policy_state.params)
                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    global_step = combined_learning_iteration_step * self.batch_size
                    mpo_aux_active = jnp.float32(global_step >= self.mpo_aux_start_step)
                    if self.mpo_aux_end_step > 0:
                        mpo_aux_active *= jnp.float32(global_step < self.mpo_aux_end_step)

                    def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b,
                                mpo_aux_q_critic_params=None, mpo_aux_actions_b=None, mpo_aux_weights_b=None):
                        # Policy loss
                        action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                        action_std = jnp.exp(action_logstd)
                        new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                        new_log_prob = new_log_prob.sum(1)
                        entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                        
                        logratio = new_log_prob - log_prob_b
                        ratio = jnp.exp(logratio)
                        approx_kl_div = (ratio - 1) - logratio
                        clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                        pg_loss1 = -advantage_b * ratio
                        pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2)
                        
                        entropy_loss = entropy.sum(1)
                        
                        # Critic loss
                        if self.nr_value_samples > 0:
                            new_value = self.critic.apply(critic_params, state_b, action_b)
                        else:
                            new_value = self.critic.apply(critic_params, state_b)
                        critic_loss = 0.5 * (new_value - return_b) ** 2

                        # Combine losses
                        loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                        if self.use_mpo_aux:
                            if self.use_mpo_aux_replay:
                                mpo_aux_q_loss = jnp.zeros_like(return_b)
                            else:
                                mpo_aux_q_value = self.mpo_aux_q_critic.apply(mpo_aux_q_critic_params, state_b, action_b).squeeze(-1)
                                mpo_aux_q_loss = 0.5 * (mpo_aux_q_value - return_b) ** 2
                            mpo_aux_action_mean, mpo_aux_action_logstd = self.policy.apply(policy_params, state_b)
                            mpo_aux_action_std = jnp.exp(mpo_aux_action_logstd)
                            mpo_aux_log_probs = -0.5 * ((mpo_aux_actions_b - mpo_aux_action_mean) / mpo_aux_action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - mpo_aux_action_logstd
                            mpo_aux_log_probs = mpo_aux_log_probs.sum(-1)
                            mpo_aux_loss = -jnp.sum(mpo_aux_weights_b * mpo_aux_log_probs)
                            loss = loss + (1.0 - jnp.float32(self.use_mpo_aux_replay)) * self.mpo_aux_q_critic_coef * mpo_aux_q_loss + mpo_aux_active * self.mpo_aux_loss_coefficient * mpo_aux_loss

                        # Create metrics
                        metrics = {
                            "loss/policy_gradient_loss": pg_loss,
                            "loss/critic_loss": critic_loss,
                            "loss/entropy_loss": entropy_loss,
                            "policy_ratio/approx_kl": approx_kl_div,
                            "policy_ratio/clip_fraction": clip_fraction,
                        }

                        if self.use_mpo_aux:
                            metrics["loss/mpo_aux_loss"] = mpo_aux_loss
                            metrics["loss/mpo_aux_q_loss"] = mpo_aux_q_loss
                            metrics["mpo_aux/weight_effective_samples"] = 1.0 / jnp.sum(mpo_aux_weights_b ** 2)
                            metrics["mpo_aux/active"] = mpo_aux_active

                        return loss, (metrics)
                    

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_log_probs = log_probs.reshape(-1)

                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    if self.use_mpo_aux:
                        vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, None, 0, 0), out_axes=0)
                        mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                        if self.use_mpo_aux_replay:
                            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)
                        else:
                            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1, 7), has_aux=True)
                    else:
                        vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                        mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
                        grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    if self.use_mpo_aux_replay:
                        def mpo_aux_replay_q_loss_fn(q_params, target_q_params, policy_params, replay_states, replay_next_states,
                                                     replay_actions, replay_rewards, replay_terminations, next_action_key):
                            next_action_mean, next_action_logstd = self.policy.apply(policy_params, replay_next_states)
                            next_actions = next_action_mean + jnp.exp(next_action_logstd) * jax.random.normal(next_action_key, shape=next_action_mean.shape)
                            target_q_values = self.mpo_aux_q_critic.apply(target_q_params, replay_next_states, next_actions).squeeze(-1)
                            target_q = replay_rewards + self.gamma * (1.0 - replay_terminations) * jnp.min(target_q_values, axis=0)
                            q_values = self.mpo_aux_q_critic.apply(q_params, replay_states, replay_actions).squeeze(-1)
                            q_loss = 0.5 * jnp.mean((q_values - jax.lax.stop_gradient(target_q)[None, :]) ** 2)
                            return q_loss, {
                                "loss/mpo_aux_replay_q_loss": q_loss,
                                "mpo_aux/replay_q_target_mean": jnp.mean(target_q),
                                "mpo_aux/replay_q_disagreement": jnp.mean(jnp.max(q_values, axis=0) - jnp.min(q_values, axis=0)),
                            }

                        grad_mpo_aux_replay_q_loss_fn = jax.value_and_grad(mpo_aux_replay_q_loss_fn, argnums=0, has_aux=True)

                    def minibatch_update(carry, minibatch_indices):
                        if self.use_mpo_aux:
                            policy_state, critic_state, mpo_aux_q_critic_state, key = carry
                        else:
                            policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        if self.use_mpo_aux:
                            key, mpo_aux_action_key = jax.random.split(key)
                            mpo_aux_states = batch_states[minibatch_indices]
                            mpo_aux_behavior_mean, mpo_aux_behavior_logstd = self.policy.apply(behavior_policy_params, mpo_aux_states)
                            mpo_aux_behavior_std = jnp.exp(mpo_aux_behavior_logstd)
                            mpo_aux_actions = mpo_aux_behavior_mean[:, None, :] + mpo_aux_behavior_std[:, None, :] * jax.random.normal(
                                mpo_aux_action_key,
                                shape=(self.minibatch_size, self.mpo_aux_action_samples) + self.as_shape,
                            )
                            mpo_aux_q_values = self.mpo_aux_q_critic.apply(
                                mpo_aux_q_critic_state.target_params if self.use_mpo_aux_replay else mpo_aux_q_critic_state.params,
                                jnp.repeat(mpo_aux_states[:, None, :], self.mpo_aux_action_samples, axis=1),
                                mpo_aux_actions,
                            ).squeeze(-1)
                            if self.use_mpo_aux_replay:
                                metrics_q_disagreement = jnp.mean(jnp.max(mpo_aux_q_values, axis=0) - jnp.min(mpo_aux_q_values, axis=0))
                                mpo_aux_q_values = jnp.min(mpo_aux_q_values, axis=0)
                            mpo_aux_normalized_q_values = (mpo_aux_q_values - jnp.mean(mpo_aux_q_values, axis=1, keepdims=True)) / (jnp.std(mpo_aux_q_values, axis=1, keepdims=True) + 1e-6)
                            mpo_aux_weights = jax.lax.stop_gradient(jax.nn.softmax(mpo_aux_normalized_q_values / self.mpo_aux_temperature, axis=1))
                            mpo_aux_actions = jax.lax.stop_gradient(mpo_aux_actions)
                            if self.use_mpo_aux_replay:
                                (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                                    policy_state.params,
                                    critic_state.params,
                                    mpo_aux_states,
                                    batch_actions[minibatch_indices],
                                    batch_log_probs[minibatch_indices],
                                    batch_returns[minibatch_indices],
                                    minibatch_advantages,
                                    mpo_aux_q_critic_state.params,
                                    mpo_aux_actions,
                                    mpo_aux_weights,
                                )
                            else:
                                (loss, (metrics)), (policy_gradients, critic_gradients, mpo_aux_q_critic_gradients) = grad_loss_fn(
                                    policy_state.params,
                                    critic_state.params,
                                    mpo_aux_states,
                                    batch_actions[minibatch_indices],
                                    batch_log_probs[minibatch_indices],
                                    batch_returns[minibatch_indices],
                                    minibatch_advantages,
                                    mpo_aux_q_critic_state.params,
                                    mpo_aux_actions,
                                    mpo_aux_weights,
                                )
                            metrics["mpo_aux/q_action_range_mean"] = jnp.mean(jnp.max(mpo_aux_q_values, axis=1) - jnp.min(mpo_aux_q_values, axis=1))
                            if self.use_mpo_aux_replay:
                                metrics["mpo_aux/target_q_disagreement_mean"] = metrics_q_disagreement
                        else:
                            (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                                policy_state.params,
                                critic_state.params,
                                batch_states[minibatch_indices],
                                batch_actions[minibatch_indices],
                                batch_log_probs[minibatch_indices],
                                batch_returns[minibatch_indices],
                                minibatch_advantages
                            )

                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        if self.use_target_q_critic:
                            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.critic_tau))

                        if self.use_mpo_aux and not self.use_mpo_aux_replay:
                            mpo_aux_q_critic_state = mpo_aux_q_critic_state.apply_gradients(grads=mpo_aux_q_critic_gradients)

                        if self.use_mpo_aux_replay:
                            key, replay_time_key, replay_env_key, replay_next_action_key = jax.random.split(key, 4)
                            replay_time_indices = jax.random.randint(replay_time_key, (self.mpo_aux_replay_batch_size,), 0, mpo_aux_replay_buffer["size"])
                            replay_env_indices = jax.random.randint(replay_env_key, (self.mpo_aux_replay_batch_size,), 0, self.nr_envs)
                            (_, replay_metrics), mpo_aux_q_critic_gradients = grad_mpo_aux_replay_q_loss_fn(
                                mpo_aux_q_critic_state.params,
                                mpo_aux_q_critic_state.target_params,
                                behavior_policy_params,
                                mpo_aux_replay_buffer["states"][replay_time_indices, replay_env_indices],
                                mpo_aux_replay_buffer["next_states"][replay_time_indices, replay_env_indices],
                                mpo_aux_replay_buffer["actions"][replay_time_indices, replay_env_indices],
                                mpo_aux_replay_buffer["rewards"][replay_time_indices, replay_env_indices],
                                mpo_aux_replay_buffer["terminations"][replay_time_indices, replay_env_indices],
                                replay_next_action_key,
                            )
                            mpo_aux_q_critic_state = mpo_aux_q_critic_state.apply_gradients(grads=mpo_aux_q_critic_gradients)
                            mpo_aux_q_critic_state = mpo_aux_q_critic_state.replace(target_params=optax.incremental_update(
                                mpo_aux_q_critic_state.params,
                                mpo_aux_q_critic_state.target_params,
                                self.mpo_aux_replay_critic_tau,
                            ))
                            metrics.update(replay_metrics)
                            metrics["mpo_aux/replay_size_fraction"] = mpo_aux_replay_buffer["size"] / self.mpo_aux_replay_buffer_size_per_env

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                        if self.use_mpo_aux:
                            metrics["gradients/mpo_aux_q_critic_grad_norm"] = optax.global_norm(mpo_aux_q_critic_gradients)

                        if self.use_mpo_aux:
                            carry = (policy_state, critic_state, mpo_aux_q_critic_state, key)
                        else:
                            carry = (policy_state, critic_state)

                        return carry, (metrics)
                    
                    if self.use_mpo_aux:
                        init_carry = (policy_state, critic_state, mpo_aux_q_critic_state, key)
                    else:
                        init_carry = (policy_state, critic_state)
                    carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
                    if self.use_mpo_aux:
                        policy_state, critic_state, mpo_aux_q_critic_state, key = carry
                    else:
                        policy_state, critic_state = carry

                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    optimization_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))
                    if self.nr_value_samples > 0:
                        optimization_metrics["q_value/sampled_action_range_mean"] = jnp.mean(sampled_action_q_ranges)
                        optimization_metrics["q_value/executed_action_advantage_abs_mean"] = jnp.mean(jnp.abs(executed_action_advantages))
                        if self.nr_q_critics > 1:
                            optimization_metrics["q_value/twin_disagreement_mean"] = jnp.mean(twin_q_disagreements)


                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    combined_metrics = tree.map_structure(lambda x: jnp.mean(x), combined_metrics)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id = carry
                        current_time = time.time()
                        metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                        is_last_train_update_before_eval = self.evaluation_active and (learning_iteration_step + 1 == self.nr_updates_per_multi_learning_iteration)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging(wandb_commit=not is_last_train_update_before_eval)

                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id))

                    if self.use_mpo_aux:
                        return (policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key), None
                    return (policy_state, critic_state, env_state, key), None
                    
                key, subkey = jax.random.split(key)
                if self.use_mpo_aux:
                    initial_learning_iteration_carry = (policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, subkey)
                else:
                    initial_learning_iteration_carry = (policy_state, critic_state, env_state, subkey)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, initial_learning_iteration_carry, jnp.arange(self.nr_updates_per_multi_learning_iteration))
                if self.use_mpo_aux:
                    policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key = learning_iteration_carry
                else:
                    policy_state, critic_state, env_state, key = learning_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        eval_action_mean, _ = self.policy.apply(policy_state.params, eval_env_state.next_observation)
                        eval_action = eval_action_mean
                        eval_processed_action = self.get_processed_action(eval_action)
                        eval_env_state = self.eval_env.step(eval_env_state, eval_processed_action)

                        return (policy_state, eval_env_state), None

                    key, reset_key = jax.random.split(key)
                    reset_keys = jax.random.split(reset_key, self.nr_envs)
                    eval_env_state = self.eval_env.reset(reset_keys, True)
                    single_eval_rollout_carry, _ = jax.lax.scan(single_eval_rollout, (policy_state, eval_env_state), jnp.arange(self.horizon))
                    _, eval_env_state = single_eval_rollout_carry

                    eval_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def callback(metrics_and_global_step):
                        metrics, combined_learning_iteration_step = metrics_and_global_step
                        global_step = int(combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs)
                        self.start_logging(global_step)
                        for key, value in metrics.items():
                            self.log(f"{key}", np.asarray(value), global_step)
                        self.end_logging()

                    combined_learning_iteration_step = (multi_learning_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration
                    jax.debug.callback(callback, (eval_metrics, combined_learning_iteration_step))
                

                # Saving
                if self.save_model:
                    if self.use_mpo_aux:
                        def save_with_check(policy_state, critic_state, mpo_aux_q_critic_state):
                            self.save(policy_state, critic_state, mpo_aux_q_critic_state)
                        jax.debug.callback(save_with_check, policy_state, critic_state, mpo_aux_q_critic_state)
                    else:
                        def save_with_check(policy_state, critic_state):
                            self.save(policy_state, critic_state)
                        jax.debug.callback(save_with_check, policy_state, critic_state)

                if self.use_mpo_aux:
                    return (policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key), None
                return (policy_state, critic_state, env_state, key), None

            if self.use_mpo_aux:
                initial_multi_learning_carry = (policy_state, critic_state, mpo_aux_q_critic_state, mpo_aux_replay_buffer, env_state, key)
            else:
                initial_multi_learning_carry = (policy_state, critic_state, env_state, key)
            jax.lax.scan(multi_learning_and_eval_save_iteration, initial_multi_learning_carry, jnp.arange(self.nr_multi_learning_and_eval_save_iterations))
            

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
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


    def save(self, policy_state, critic_state, mpo_aux_q_critic_state=None):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state
        }
        if self.use_mpo_aux:
            checkpoint["mpo_aux_q_critic"] = mpo_aux_q_critic_state
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
        model = PPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        if model.use_mpo_aux:
            target["mpo_aux_q_critic"] = model.mpo_aux_q_critic_state
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        if model.use_mpo_aux:
            model.mpo_aux_q_critic_state = checkpoint["mpo_aux_q_critic"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state, key):
            # key, subkey = jax.random.split(key)
            action_mean, action_logstd = self.policy.apply(self.policy_state.params, env_state.next_observation)
            # action_std = jnp.exp(action_logstd)
            action = action_mean # + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            processed_action = self.get_processed_action(action)
            env_state = self.train_env.step(env_state, processed_action)
            return env_state, key

        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        env_state = self.train_env.reset(reset_keys, True)
        while True:
            env_state, self.key = rollout(env_state, self.key)
            if self.render:
                env_state = self.train_env.render(env_state)


    def general_properties():
        return GeneralProperties
