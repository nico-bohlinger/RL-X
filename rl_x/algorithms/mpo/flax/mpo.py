import os
import shutil
import json
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.mpo.flax.general_properties import GeneralProperties
from rl_x.algorithms.mpo.flax.policy import get_policy
from rl_x.algorithms.mpo.flax.critic import get_critic
from rl_x.algorithms.mpo.flax.dual_variables import DualVariables
from rl_x.algorithms.mpo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.mpo.flax.rl_train_state import PolicyTrainState, CriticTrainState

rlx_logger = logging.getLogger("rl_x")


class MPO():
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
        self.init_log_eta = config.algorithm.init_log_eta
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.init_log_penalty_temperature = config.algorithm.init_log_penalty_temperature
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

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, dual_key = jax.random.split(self.key, 4)

        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        nr_actions = np.prod(self.train_env.single_action_space.shape).item()
        self.dual_variables = DualVariables(nr_actions, self.init_log_eta, self.init_log_alpha_mean, self.init_log_alpha_stddev, self.init_log_penalty_temperature)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.dual_variables.apply = jax.jit(self.dual_variables.apply)

        def agent_linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.agent_learning_rate * fraction
        
        def dual_linear_schedule(count):
            step = (count * self.nr_envs) - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / total_steps)
            return self.dual_learning_rate * fraction

        self.policy_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        self.critic_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        self.dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])
        action = jnp.array([self.train_env.single_action_space.sample()])

        self.policy_state = PolicyTrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            target_params=self.policy.init(policy_key, state),
            env_params=self.policy.init(policy_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate),
            )
        )

        self.critic_state = CriticTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_learning_rate),
            )
        )

        self.dual_variables_state = TrainState.create(
            apply_fn=self.dual_variables.apply,
            params=self.dual_variables.init(dual_key),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.dual_learning_rate),
            )
        )

        if self.enable_observation_normalization:
            self.observation_normalizer_state = {
                "running_mean": np.zeros((1, state.shape[-1]), dtype=np.float32),
                "running_var": np.ones((1, state.shape[-1]), dtype=np.float32),
                "running_std_dev": np.ones((1, state.shape[-1]), dtype=np.float32),
                "count": 0
            }
        else:
            self.observation_normalizer_state = None

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def _normalize_observations(self, observations, update=False, no_normalize_after_update=True):
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

            if no_normalize_after_update:
                return

        return (observations - self.observation_normalizer_state["running_mean"]) / (self.observation_normalizer_state["running_std_dev"] + self.float_epsilon)
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: PolicyTrainState, state: np.ndarray, key: jax.random.PRNGKey):
            action_mean, action_std = self.policy.apply(policy_state.env_params, state)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            return action, key
        

        @jax.jit
        def update_actor(policy_state: PolicyTrainState):
            return policy_state.replace(env_params=policy_state.target_params)
        

        @jax.jit
        def update_target_networks(policy_state: PolicyTrainState, critic_state: CriticTrainState):
            return policy_state.replace(target_params=policy_state.params), critic_state.replace(target_params=critic_state.params)
        

        @jax.jit
        def update(
            policy_state: PolicyTrainState, critic_state: CriticTrainState, dual_variables_state: TrainState,
            states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, truncations: np.ndarray, all_effective_n_steps: np.ndarray, key: jax.random.PRNGKey
        ):
            def loss_fn(policy_params, policy_target_params, critic_params, critic_target_params, dual_variables_params, state, next_state, action, reward, done, truncation, effective_n_step, key1, key2):
                # Critic update
                target_next_action_mean, target_next_action_std = self.policy.apply(stop_gradient(policy_target_params), next_state)
                sampled_next_actions = target_next_action_mean[None, :] + target_next_action_std[None, :] * jax.random.normal(key1, shape=(self.action_sampling_number,) + target_next_action_mean.shape)  # (sampled actions, action_dim)

                expanded_next_states = jnp.repeat(next_state[None, :], self.action_sampling_number, axis=0)  # (sampled actions, state_dim)
                target_next_logits = self.critic.apply(stop_gradient(critic_target_params), expanded_next_states, sampled_next_actions)  # (sampled actions, nr_atoms)
                target_next_pmf = jax.nn.softmax(target_next_logits, axis=-1)[:, None, :]  # (sampled actions, 1, nr_atoms)

                bootstrap = 1.0 - (done * (1.0 - truncation))
                discount = (self.gamma ** effective_n_step) * bootstrap
                q_support = jnp.linspace(self.v_min, self.v_max, self.nr_atoms)
                target_z = jnp.clip(reward + discount * q_support, self.v_min, self.v_max)  # (atoms,)

                abs_delta = jnp.abs(target_z[None, None, :] - q_support[None, :, None])  # (1, atoms, 1)

                delta_z = q_support[1] - q_support[0]
                target_pmf = jnp.clip(1.0 - abs_delta / delta_z, 0.0, 1.0) * target_next_pmf  # (sampled actions, atoms, atoms)
                target_pmf = jnp.sum(target_pmf, axis=-1)  # (sampled actions, atoms)
                target_pmf = target_pmf.mean(0)  # (atoms,)

                current_logits = self.critic.apply(critic_params, state, action)  # (nr_atoms,)
                current_q = jnp.sum(jax.nn.softmax(current_logits, axis=-1) * q_support, axis=-1)  # ()
                q_loss = -jnp.sum(target_pmf * jax.nn.log_softmax(current_logits, axis=-1), axis=-1)

                # Actor and dual update
                stacked_states = jnp.concatenate([state[None, :], next_state[None, :]], axis=0)  # (2, state_dim)
                target_action_mean, target_action_std = self.policy.apply(stop_gradient(policy_target_params), stacked_states)
                sampled_actions = target_action_mean[None, :, :] + target_action_std[None, :, :] * jax.random.normal(key2, shape=(self.action_sampling_number,) + target_action_mean.shape)  # (sampled actions, 2, action_dim)

                expanded_states = jnp.repeat(stacked_states[None, :, :], self.action_sampling_number, axis=0)  # (sampled actions, 2, state_dim)
                expanded_stacked_logits = self.critic.apply(stop_gradient(critic_target_params), expanded_states.reshape((-1, expanded_states.shape[-1])), sampled_actions.reshape((-1, sampled_actions.shape[-1])))  # (sampled actions * 2, nr_atoms)
                expanded_stacked_logits = expanded_stacked_logits.reshape((self.action_sampling_number, 2, -1))  # (sampled actions, 2, nr_atoms)
                
                expanded_stacked_pmf = jax.nn.softmax(expanded_stacked_logits, axis=-1)  # (sampled actions, 2, nr_atoms)
                expanded_stacked_q = jnp.sum(expanded_stacked_pmf * q_support, axis=-1)  # (sampled actions, 2)

                log_eta, log_alpha_mean, log_alpha_stddev, log_penalty_temperature = self.dual_variables.apply(dual_variables_params)
                eta = jax.nn.softplus(log_eta) + self.float_epsilon
                improvement_dist = jax.nn.softmax(expanded_stacked_q / stop_gradient(eta), axis=0)  # (sampled actions, 2)

                q_logsumexp = jax.scipy.special.logsumexp(expanded_stacked_q / eta, axis=0)  # (2,)
                loss_eta = eta * (self.epsilon_non_parametric + jnp.mean(q_logsumexp) - jnp.log(self.action_sampling_number))

                if self.action_clipping:
                    penalty_temperature = jax.nn.softplus(log_penalty_temperature) + self.float_epsilon
                    diff_oob = sampled_actions - jnp.clip(sampled_actions, -1.0, 1.0)  # (sampled actions, 2, action_dim)
                    cost_oob = -jnp.linalg.norm(diff_oob, axis=-1)  # (sampled actions, 2)
                    penalty_improvement = jax.nn.softmax(cost_oob / stop_gradient(penalty_temperature), axis=0)

                    penalty_logsumexp = jax.scipy.special.logsumexp(cost_oob / penalty_temperature, axis=0)  # (2,)
                    loss_penalty_temp = penalty_temperature * (self.epsilon_penalty + jnp.mean(penalty_logsumexp) - jnp.log(self.action_sampling_number))

                    improvement_dist = improvement_dist + penalty_improvement
                    loss_eta = loss_eta + loss_penalty_temp

                    penalty_temperature_detached = stop_gradient(penalty_temperature)
                else:
                    penalty_temperature_detached = jnp.array(0.0, dtype=jnp.float32)

                online_action_mean, online_action_std =  self.policy.apply(policy_params, stacked_states)

                alpha_mean = jax.nn.softplus(log_alpha_mean) + self.float_epsilon
                alpha_std = (jnp.logaddexp(log_alpha_stddev, jnp.zeros_like(log_alpha_stddev)) + self.float_epsilon)

                logprob_mean = jnp.sum(-0.5 * ((((sampled_actions - online_action_mean) / target_action_std) ** 2) + jnp.log(2.0 * jnp.pi)) - jnp.log(target_action_std), axis=-1)  # (sampled actions, 2 * batch)

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
                loss = q_loss + actor_loss + dual_loss
                
                # Create metrics
                metrics = {
                    "loss/critic_loss": q_loss,
                    "loss/actor_loss": actor_loss,
                    "loss/dual_loss": dual_loss,
                    "q/current_q_mean": current_q,
                    "dual/eta": eta,
                    "dual/penalty_temperature": penalty_temperature_detached,
                    "dual/alpha_mean": alpha_mean.mean(),
                    "dual/alpha_std": alpha_std.mean(),
                    "loss/loss_eta": loss_eta,
                    "loss/loss_alpha": loss_alpha_mean + loss_alpha_std,
                    "kl/mean_kl_mean": mean_kl_mean.mean(),
                    "kl/mean_kl_std": mean_kl_std.mean(),
                    "policy/std_min_mean": jnp.min(online_action_std, axis=1).mean(),
                    "policy/std_max_mean": jnp.max(online_action_std, axis=1).mean(),
                }

                return loss, metrics

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_loss_fn = lambda *a, **k: jax.tree_util.tree_map(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_loss_fn, argnums=(0, 2, 4), has_aux=True)

            key, subkeys1, subkeys2 = jax.random.split(key, 3)
            per_sample_keys1 = jax.random.split(subkeys1, states.shape[0])
            per_sample_keys2 = jax.random.split(subkeys2, states.shape[0])

            (loss, metrics), (policy_grads, critic_grads, dual_variables_grads) = grad_loss_fn(policy_state.params, policy_state.target_params, critic_state.params, critic_state.target_params, dual_variables_state.params, states, next_states, actions, rewards, dones, truncations, all_effective_n_steps, per_sample_keys1, per_sample_keys2)
            
            policy_state = policy_state.apply_gradients(grads=policy_grads)
            critic_state = critic_state.apply_gradients(grads=critic_grads)
            dual_variables_state = dual_variables_state.apply_gradients(grads=dual_variables_grads)

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
            
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grads)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grads)
            metrics["gradients/dual_variables_grad_norm"] = optax.global_norm(dual_variables_grads)

            return policy_state, critic_state, dual_variables_state, metrics, key


        @jax.jit
        def get_deterministic_action(policy_state: PolicyTrainState, state: np.ndarray):
            action_mean, _ = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(max(1, self.buffer_size // self.nr_envs)), self.nr_envs, self.train_env.single_observation_space.shape, self.train_env.single_action_space.shape, self.n_steps, self.gamma, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
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
            if logging_time_prev:
                time_metrics_collection.setdefault("time/logging_time_prev", []).append(logging_time_prev)


            # Acting
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                processed_action = np.array([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                normalized_state = self._normalize_observations(state, update=False)
                action, self.key = get_action(self.policy_state, normalized_state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)

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
                self.policy_state = update_actor(self.policy_state)


            # Optimizing
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, batch_truncation, batch_effective_n_steps = replay_buffer.sample(self.batch_size)
                if self.enable_observation_normalization:
                    combined_states = np.concatenate([batch_states, batch_next_states], axis=0)
                    self._normalize_observations(combined_states, update=True, no_normalize_after_update=True)
                batch_normalized_states = self._normalize_observations(batch_states, update=False)
                batch_normalized_next_states = self._normalize_observations(batch_next_states, update=False)

                self.policy_state, self.critic_state, self.dual_variables_state, optimization_metrics, self.key = update(
                    self.policy_state,
                    self.critic_state,
                    self.dual_variables_state,
                    batch_normalized_states,
                    batch_normalized_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_dones,
                    batch_truncation,
                    batch_effective_n_steps,
                    self.key
                )
                nr_updates += 1

                if nr_updates % self.target_network_update_period == 0:
                    self.policy_state, self.critic_state = update_target_networks(self.policy_state, self.critic_state)

                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    eval_normalized_state = self._normalize_observations(eval_state, update=False)
                    eval_processed_action = get_deterministic_action(self.policy_state, eval_normalized_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_processed_action))
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


    def get_buffer_mean(self, buffer):
        if len(buffer) > 0:
            return np.mean(buffer)
        else:
            return 0.0


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
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state,
            "dual_variables": self.dual_variables_state,
            "observation_normalizer": self.observation_normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.best_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.best_model_file_name}.zip", f"{self.save_path}/{self.best_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


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
        model = MPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "dual_variables": model.dual_variables_state,
            "observation_normalizer": model.observation_normalizer_state,
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
        @jax.jit
        def get_action(policy_state: PolicyTrainState, state: np.ndarray):
            action_mean, _ = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                normalized_state = self._normalize_observations(state, update=False)
                processed_action = get_action(self.policy_state, normalized_state)
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
            return GeneralProperties
