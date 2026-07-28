import os
import shutil
import json
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

from rl_x.algorithms.fastmpo.flax.general_properties import GeneralProperties
from rl_x.algorithms.fastmpo.flax.policy import get_policy
from rl_x.algorithms.fastmpo.flax.critic import get_critic
from rl_x.algorithms.fastmpo.flax.dual_variables import DualVariables
from rl_x.algorithms.fastmpo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.fastmpo.flax.rl_train_state import RLTrainState

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
        self.critic_network_type = config.algorithm.critic_network_type
        self.dual_critic = config.algorithm.dual_critic
        self.policy_network_type = config.algorithm.policy_network_type
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
        self.horizon = self.train_env.horizon

        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")
        if self.evaluation_and_save_frequency != -1 and self.evaluation_and_save_frequency % self.nr_envs != 0:
            raise ValueError("The evaluation and save frequency must be a multiple of the number of environments.")
        if self.learning_starts < self.n_steps:
            raise ValueError("The replay buffer must contain at least n_steps transitions before learning starts.")
        if self.nr_parallel_seeds != 1:
            raise ValueError("Parallel seeds are only supported by the fully JIT-compiled implementation.")
        if self.clipped_double_q_learning and not self.dual_critic:
            raise ValueError("Clipped double Q-learning requires two critics.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, dual_key = jax.random.split(self.key, 4)
        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        nr_actions = np.prod(self.train_env.single_action_space.shape).item()
        self.dual_variables = DualVariables(nr_actions, config.algorithm.init_log_eta, config.algorithm.init_log_alpha_mean, config.algorithm.init_log_alpha_stddev, config.algorithm.init_log_penalty_temperature)
        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.dual_variables.apply = jax.jit(self.dual_variables.apply)

        def policy_linear_schedule(count):
            return self.policy_learning_rate * (1.0 - count * self.nr_envs / self.total_timesteps)

        def critic_linear_schedule(count):
            return self.critic_learning_rate * (1.0 - count * self.nr_envs / self.total_timesteps)

        def dual_linear_schedule(count):
            return self.dual_learning_rate * (1.0 - count * self.nr_envs / self.total_timesteps)

        policy_learning_rate = policy_linear_schedule if self.anneal_policy_learning_rate else self.policy_learning_rate
        critic_learning_rate = critic_linear_schedule if self.anneal_critic_learning_rate else self.critic_learning_rate
        dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate
        state = jnp.array([self.train_env.single_observation_space.sample()])
        action = jnp.zeros((1,) + self.train_env.single_action_space.shape, dtype=jnp.float32)
        policy_params = self.policy.init(policy_key, state)
        critic_params = self.critic.init(critic_key, state, action)

        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=policy_params,
            target_params=policy_params,
            tx=optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.inject_hyperparams(optax.adamw)(learning_rate=policy_learning_rate, weight_decay=self.policy_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2)),
        )
        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            target_params=critic_params,
            tx=optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.inject_hyperparams(optax.adamw)(learning_rate=critic_learning_rate, weight_decay=self.critic_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2)),
        )
        self.dual_variables_state = TrainState.create(
            apply_fn=self.dual_variables.apply,
            params=self.dual_variables.init(dual_key),
            tx=optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.inject_hyperparams(optax.adamw)(learning_rate=dual_learning_rate, weight_decay=self.dual_weight_decay, b1=self.adam_beta1, b2=self.adam_beta2)),
        )

        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": np.zeros((1, state.shape[-1]), dtype=np.float32),
                "running_var": np.ones((1, state.shape[-1]), dtype=np.float32),
                "running_std_dev": np.ones((1, state.shape[-1]), dtype=np.float32),
                "count": 0,
            }
        else:
            self.observation_normalizer_state = None

        if self.save_model:
            os.makedirs(self.save_path, exist_ok=True)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def normalize_observations(self, observations, update=False):
        if not self.enable_observation_normalization:
            return observations
        if update:
            batch_mean = np.mean(observations, axis=0, keepdims=True)
            batch_var = np.var(observations, axis=0, keepdims=True)
            batch_count = observations.shape[0]
            new_count = self.observation_normalizer_state["count"] + batch_count
            delta = batch_mean - self.observation_normalizer_state["running_mean"]
            self.observation_normalizer_state["running_mean"] += delta * batch_count / new_count
            delta2 = batch_mean - self.observation_normalizer_state["running_mean"]
            m_a = self.observation_normalizer_state["running_var"] * self.observation_normalizer_state["count"]
            m_b = batch_var * batch_count
            m2 = m_a + m_b + np.square(delta2) * self.observation_normalizer_state["count"] * batch_count / new_count
            self.observation_normalizer_state["running_var"] = m2 / new_count
            self.observation_normalizer_state["running_std_dev"] = np.sqrt(self.observation_normalizer_state["running_var"])
            self.observation_normalizer_state["count"] = new_count
        return (observations - self.observation_normalizer_state["running_mean"]) / (self.observation_normalizer_state["running_std_dev"] + self.normalizer_epsilon)


    def train(self):
        @jax.jit
        def update_critic(policy_target_params, critic_state, normalized_states, normalized_next_states, actions, rewards, dones, truncations, effective_n_steps, key):
            def loss_fn(critic_params, normalized_state, normalized_next_state, action, reward, done, truncated, sample_n_steps, sample_key):
                next_action_mean, next_action_std = self.policy.apply(policy_target_params, normalized_next_state)
                sampled_next_actions_raw = next_action_mean[None] + next_action_std[None] * jax.random.normal(sample_key, shape=(self.action_sampling_number,) + next_action_mean.shape)
                sampled_next_actions = jnp.clip(sampled_next_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_next_actions_raw
                expanded_next_states = jnp.repeat(normalized_next_state[None], self.action_sampling_number, axis=0)
                target_next_pmf = jax.nn.softmax(self.critic.apply(critic_state.target_params, expanded_next_states, sampled_next_actions), axis=-1)
                mean_target_next_pmf = jnp.mean(target_next_pmf, axis=1)
                q_support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
                discount = self.gamma ** sample_n_steps * (1.0 - done * (1.0 - truncated))
                target_z = jnp.clip(reward + discount * q_support, self.v_min, self.v_max)
                b = (target_z - self.v_min) / ((self.v_max - self.v_min) / (self.nr_atoms - 1))
                lower = jnp.floor(b).astype(jnp.int32)
                upper = jnp.ceil(b).astype(jnp.int32)
                lower = jnp.where((upper == lower) & (lower > 0), lower - 1, lower)
                upper = jnp.where((upper == lower) & (lower == 0), upper + 1, upper)
                lower_weight = upper.astype(jnp.float32) - b
                upper_weight = b - lower.astype(jnp.float32)
                nr_critics = mean_target_next_pmf.shape[0]
                critic_indices = jnp.repeat(jnp.arange(nr_critics)[:, None], self.nr_atoms, axis=1)
                projected = jnp.zeros_like(mean_target_next_pmf)
                projected = projected.at[(critic_indices, jnp.repeat(lower[None], nr_critics, axis=0))].add(mean_target_next_pmf * lower_weight)
                projected = projected.at[(critic_indices, jnp.repeat(upper[None], nr_critics, axis=0))].add(mean_target_next_pmf * upper_weight)
                target_values = jnp.sum(projected * q_support, axis=-1)
                if self.dual_critic and self.clipped_double_q_learning:
                    chosen = jnp.where(target_values[0] <= target_values[1], projected[0], projected[1])
                    target_distribution = jnp.stack([chosen, chosen])
                else:
                    target_distribution = projected
                action = jnp.clip(action, -1.0, 1.0) if self.action_clipping else action
                current_log_pmf = jax.nn.log_softmax(self.critic.apply(critic_params, normalized_state, action), axis=-1)
                q_loss = -jnp.sum(target_distribution * current_log_pmf)
                return q_loss, {"loss/q_loss": q_loss, "q/q_mean": jnp.mean(target_values), "q/q_max": jnp.max(target_values), "q/q_min": jnp.min(target_values)}

            vmapped_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
            mean_loss_fn = lambda *args: jax.tree_util.tree_map(jnp.mean, vmapped_loss_fn(*args))
            grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=0, has_aux=True)
            key, sample_key = jax.random.split(key)
            sample_keys = jax.random.split(sample_key, normalized_states.shape[0])
            (loss, metrics), gradients = grad_loss_fn(critic_state.params, normalized_states, normalized_next_states, actions, rewards, dones, truncations, effective_n_steps, sample_keys)
            critic_state = critic_state.apply_gradients(grads=gradients)
            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.critic_tau))
            metrics["lr/critic_learning_rate"] = critic_state.opt_state[1].hyperparams["learning_rate"]
            metrics["gradients/critic_grad_norm"] = optax.global_norm(gradients)
            return critic_state, metrics, key


        @jax.jit
        def update_policy_and_dual(policy_state, critic_target_params, dual_variables_state, normalized_states, normalized_next_states, key):
            def loss_fn(policy_params, dual_variables_params, normalized_state, normalized_next_state, action_key):
                stacked_states = jnp.stack([normalized_state, normalized_next_state])
                target_action_mean, target_action_std = self.policy.apply(policy_state.target_params, stacked_states)
                sampled_actions_raw = target_action_mean[None] + target_action_std[None] * jax.random.normal(action_key, shape=(self.action_sampling_number,) + target_action_mean.shape)
                sampled_actions = jnp.clip(sampled_actions_raw, -1.0, 1.0) if self.action_clipping else sampled_actions_raw
                expanded_states = jnp.repeat(stacked_states[None], self.action_sampling_number, axis=0)
                logits = self.critic.apply(critic_target_params, expanded_states.reshape((-1, expanded_states.shape[-1])), sampled_actions.reshape((-1, sampled_actions.shape[-1])))
                q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * jnp.linspace(self.v_min, self.v_max, self.nr_atoms), axis=-1)
                q_values = q_values.reshape((q_values.shape[0], self.action_sampling_number, 2))
                if q_values.shape[0] == 1:
                    q_values = q_values[0]
                elif self.clipped_double_q_learning:
                    q_values = jnp.min(q_values, axis=0)
                else:
                    q_values = jnp.mean(q_values, axis=0)

                log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature = self.dual_variables.apply(dual_variables_params)
                eta = jax.nn.softplus(log_eta)[0] + self.float_epsilon
                improvement_distribution = jax.nn.softmax(q_values / stop_gradient(eta), axis=0)
                loss_eta = eta * (self.epsilon_non_parametric + jnp.mean(jax.scipy.special.logsumexp(q_values / eta, axis=0)) - jnp.log(self.action_sampling_number))

                if self.action_clipping:
                    penalty_temperature = jax.nn.softplus(log_penalty_temperature)[0] + self.float_epsilon
                    out_of_bounds_cost = -jnp.linalg.norm(sampled_actions_raw - jnp.clip(sampled_actions_raw, -1.0, 1.0), axis=-1)
                    improvement_distribution += jax.nn.softmax(out_of_bounds_cost / stop_gradient(penalty_temperature), axis=0)
                    loss_eta += penalty_temperature * (self.epsilon_penalty + jnp.mean(jax.scipy.special.logsumexp(out_of_bounds_cost / penalty_temperature, axis=0)) - jnp.log(self.action_sampling_number))
                else:
                    penalty_temperature = jnp.float32(0.0)

                online_action_mean, online_action_std = self.policy.apply(policy_params, stacked_states)
                alpha_mean = jax.nn.softplus(log_alpha_mean) + self.float_epsilon
                alpha_std = jax.nn.softplus(log_alpha_stddev) + self.float_epsilon
                logprob_mean = jnp.sum(-0.5 * (((sampled_actions_raw - online_action_mean) / target_action_std) ** 2 + jnp.log(2.0 * jnp.pi)) - jnp.log(target_action_std), axis=-1)
                loss_pg_mean = -jnp.mean(jnp.sum(logprob_mean * improvement_distribution, axis=0))
                target_action_std_clipped = jnp.clip(target_action_std, min=self.float_epsilon)
                kl_mean = (target_action_mean - online_action_mean) ** 2 / (2.0 * target_action_std_clipped ** 2)
                mean_kl_mean = jnp.mean(kl_mean, axis=0)
                loss_kl_mean = jnp.sum(stop_gradient(alpha_mean) * mean_kl_mean)
                loss_alpha_mean = jnp.sum(alpha_mean * (self.epsilon_parametric_mu - stop_gradient(mean_kl_mean)))

                logprob_std = jnp.sum(-0.5 * (((sampled_actions_raw - target_action_mean) / online_action_std) ** 2 + jnp.log(2.0 * jnp.pi)) - jnp.log(online_action_std), axis=-1)
                loss_pg_std = -jnp.mean(jnp.sum(logprob_std * improvement_distribution, axis=0))
                online_action_std_clipped = jnp.clip(online_action_std, min=self.float_epsilon)
                kl_std = jnp.log(online_action_std_clipped / target_action_std_clipped) + target_action_std_clipped ** 2 / (2.0 * online_action_std_clipped ** 2) - 0.5
                mean_kl_std = jnp.mean(kl_std, axis=0)
                loss_kl_std = jnp.sum(stop_gradient(alpha_std) * mean_kl_std)
                loss_alpha_std = jnp.sum(alpha_std * (self.epsilon_parametric_sigma - stop_gradient(mean_kl_std)))

                actor_loss = loss_pg_mean + loss_pg_std + loss_kl_mean + loss_kl_std
                dual_loss = loss_alpha_mean + loss_alpha_std + loss_eta
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
                    "dual/eta": eta,
                    "dual/penalty_temperature": penalty_temperature,
                    "dual/alpha_mean": jnp.mean(alpha_mean),
                    "dual/alpha_std": jnp.mean(alpha_std),
                    "kl/mean_kl_mean": jnp.mean(mean_kl_mean),
                    "kl/mean_kl_std": jnp.mean(mean_kl_std),
                    "q/improvement_q_mean": jnp.mean(q_values),
                    "policy/std_min_mean": jnp.mean(jnp.min(online_action_std, axis=-1)),
                    "policy/std_max_mean": jnp.mean(jnp.max(online_action_std, axis=-1)),
                }
                return actor_loss + dual_loss, metrics

            vmapped_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0))
            mean_loss_fn = lambda *args: jax.tree_util.tree_map(jnp.mean, vmapped_loss_fn(*args))
            grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=(0, 1), has_aux=True)
            key, action_key = jax.random.split(key)
            action_keys = jax.random.split(action_key, normalized_states.shape[0])
            (loss, metrics), (policy_gradients, dual_gradients) = grad_loss_fn(policy_state.params, dual_variables_state.params, normalized_states, normalized_next_states, action_keys)
            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            dual_variables_state = dual_variables_state.apply_gradients(grads=dual_gradients)
            dual_variables_state = dual_variables_state.replace(params={"params": {
                "log_eta": jnp.maximum(dual_variables_state.params["params"]["log_eta"], self.min_log_temperature),
                "log_alpha_mean": jnp.maximum(dual_variables_state.params["params"]["log_alpha_mean"], self.min_log_alpha),
                "log_alpha_stddev": jnp.maximum(dual_variables_state.params["params"]["log_alpha_stddev"], self.min_log_alpha),
                "log_penalty_temperature": dual_variables_state.params["params"]["log_penalty_temperature"],
            }})
            policy_state = policy_state.replace(target_params=optax.incremental_update(policy_state.params, policy_state.target_params, self.policy_tau))
            metrics["lr/policy_learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            metrics["lr/dual_variables_learning_rate"] = dual_variables_state.opt_state[1].hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/dual_variables_grad_norm"] = optax.global_norm(dual_gradients)
            return policy_state, dual_variables_state, metrics, key


        self.set_train_mode()
        replay_buffer = ReplayBuffer(self.buffer_size_per_env, self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma, self.rng)
        state, _ = self.train_env.reset()
        global_step = 0
        nr_critic_updates = 0
        nr_policy_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)

            # Acting
            dones_this_rollout = 0
            normalized_state = self.normalize_observations(state)
            self.key, action_key = jax.random.split(self.key)
            policy_params = self.policy_state.params if self.collect_data_with_online_policy else self.policy_state.target_params
            action_mean, action_std = self.policy.apply(policy_params, normalized_state)
            action = action_mean + action_std * jax.random.normal(action_key, shape=action_mean.shape)
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(self.get_processed_action(action)))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for index, single_done in enumerate(done):
                if single_done:
                    actual_next_state[index] = np.array(self.train_env.get_final_observation_at_index(info, index))
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            replay_buffer.add(state, actual_next_state, jax.device_get(action), reward, done, truncated)
            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)

            should_optimize = global_step > self.learning_starts * self.nr_envs
            should_evaluate = self.evaluation_active and self.evaluation_and_save_frequency != -1 and global_step % self.evaluation_and_save_frequency == 0
            should_save = should_optimize and self.save_model and self.evaluation_and_save_frequency != -1 and global_step % self.evaluation_and_save_frequency == 0
            should_log = global_step % self.logging_frequency == 0

            # Optimizing
            if should_optimize:
                total_critic_updates = self.nr_policy_updates_per_step * self.nr_critic_updates_per_policy_update
                sample_count = total_critic_updates * self.batch_size
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncations, batch_effective_n_steps = replay_buffer.sample(sample_count)
                if self.enable_observation_normalization:
                    self.normalize_observations(np.concatenate([batch_states, batch_next_states]), update=True)
                normalized_states = self.normalize_observations(batch_states).reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                normalized_next_states = self.normalize_observations(batch_next_states).reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_actions = batch_actions.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size, -1)
                batch_rewards = batch_rewards.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_dones = batch_dones.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_truncations = batch_truncations.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)
                batch_effective_n_steps = batch_effective_n_steps.reshape(self.nr_policy_updates_per_step, self.nr_critic_updates_per_policy_update, self.batch_size)

                for policy_update in range(self.nr_policy_updates_per_step):
                    for critic_update in range(self.nr_critic_updates_per_policy_update):
                        self.critic_state, metrics, self.key = update_critic(self.policy_state.target_params, self.critic_state, normalized_states[policy_update, critic_update], normalized_next_states[policy_update, critic_update], batch_actions[policy_update, critic_update], batch_rewards[policy_update, critic_update], batch_dones[policy_update, critic_update], batch_truncations[policy_update, critic_update], batch_effective_n_steps[policy_update, critic_update], self.key)
                        for key, value in metrics.items():
                            optimization_metrics_collection.setdefault(key, []).append(value)
                        nr_critic_updates += 1
                    self.policy_state, self.dual_variables_state, metrics, self.key = update_policy_and_dual(self.policy_state, self.critic_state.target_params, self.dual_variables_state, normalized_states[policy_update, -1], normalized_next_states[policy_update, -1], self.key)
                    for key, value in metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    nr_policy_updates += 1
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)

            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                for _ in range(self.horizon):
                    eval_action_mean, _ = self.policy.apply(self.policy_state.params, self.normalize_observations(eval_state))
                    eval_state, _, _, _, eval_info = self.eval_env.step(jax.device_get(self.get_processed_action(eval_action_mean)))
                    eval_logging_info = self.eval_env.get_logging_info_dict(eval_info)
                    if "episode_return" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_return", []).extend(eval_logging_info["episode_return"])
                    if "episode_length" in eval_logging_info:
                        evaluation_metrics_collection.setdefault("eval/episode_length", []).extend(eval_logging_info["episode_length"])
                self.set_train_mode()
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)

            # Saving
            if should_save:
                self.save()
            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            # Logging
            if should_log:
                self.start_logging(global_step)
                rollout_info_metrics = {}
                env_info_metrics = {}
                for info_name, values in step_info_collection.items():
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(values)
                    if mean_value == mean_value:
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                steps_metrics = {"steps/nr_env_steps": global_step, "steps/nr_critic_updates": nr_critic_updates, "steps/nr_policy_updates": nr_policy_updates, "steps/nr_episodes": nr_episodes}
                for key, value in {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}.items():
                    self.log(key, value, global_step)
                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}
                self.end_logging()
            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


    def log(self, name, value, step):
        if self.track_wandb:
            wandb.log({"global_step": int(step), name: value})
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
        checkpoint = {"policy": self.policy_state, "critic": self.critic_state, "dual_variables": self.dual_variables_state, "observation_normalizer": self.observation_normalizer_state}
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as file:
            json.dump(self.config.algorithm.to_dict(), file)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")
        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        split_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(split_path[:-1]))
        checkpoint_file_name = split_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"
        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FastMPO(config, train_env, eval_env, run_path, writer)
        target = {"policy": model.policy_state, "critic": model.critic_state, "dual_variables": model.dual_variables_state, "observation_normalizer": model.observation_normalizer_state}
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_dir, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.dual_variables_state = checkpoint["dual_variables"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        shutil.rmtree(checkpoint_dir)
        return model


    def test(self, episodes):
        self.set_eval_mode()
        for episode in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                action_mean, _ = self.policy.apply(self.policy_state.params, self.normalize_observations(state))
                state, reward, terminated, truncated, _ = self.eval_env.step(jax.device_get(self.get_processed_action(action_mean)))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {episode + 1} - Return: {episode_return}")


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
