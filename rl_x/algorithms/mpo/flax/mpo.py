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
import flax
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tensorflow_probability.substrates.jax.distributions

from rl_x.algorithms.mpo.flax.general_properties import GeneralProperties
from rl_x.algorithms.mpo.flax.policy import get_policy
from rl_x.algorithms.mpo.flax.critic import get_critic
from rl_x.algorithms.mpo.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.mpo.flax.types import AgentParams, DualParams, TrainingState

rlx_logger = logging.getLogger("rl_x")


class MPO():
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
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.agent_learning_rate = config.algorithm.agent_learning_rate
        self.dual_learning_rate = config.algorithm.dual_learning_rate
        self.anneal_agent_learning_rate = config.algorithm.anneal_agent_learning_rate
        self.anneal_dual_learning_rate = config.algorithm.anneal_dual_learning_rate
        self.ensemble_size = config.algorithm.ensemble_size
        self.nr_atoms_per_net = config.algorithm.nr_atoms_per_net
        self.nr_dropped_atoms_per_net = config.algorithm.nr_dropped_atoms_per_net
        self.huber_kappa = config.algorithm.huber_kappa
        self.nr_samples = config.algorithm.nr_samples
        self.stability_epsilon = config.algorithm.stability_epsilon
        self.init_log_temperature = config.algorithm.init_log_temperature
        self.init_log_alpha_mean = config.algorithm.init_log_alpha_mean
        self.init_log_alpha_stddev = config.algorithm.init_log_alpha_stddev
        self.min_log_temperature = config.algorithm.min_log_temperature
        self.min_log_alpha = config.algorithm.min_log_alpha
        self.kl_epsilon = config.algorithm.kl_epsilon
        self.kl_epsilon_penalty = config.algorithm.kl_epsilon_penalty
        self.kl_epsilon_mean = config.algorithm.kl_epsilon_mean
        self.kl_epsilon_stddev = config.algorithm.kl_epsilon_stddev
        self.retrace_lambda = config.algorithm.retrace_lambda
        self.trace_length = config.algorithm.trace_length
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.update_period = config.algorithm.update_period
        self.gamma = config.algorithm.gamma
        self.max_param_update = config.algorithm.max_param_update
        self.logging_all_metrics = config.algorithm.logging_all_metrics
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.nr_total_atoms = self.nr_atoms_per_net * self.ensemble_size
        self.nr_target_atoms = self.nr_total_atoms - (self.nr_dropped_atoms_per_net * self.ensemble_size)

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.env_as_low = env.single_action_space.low
        self.env_as_high = env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def agent_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps))
            return self.agent_learning_rate * fraction
        
        def dual_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps))
            return self.dual_learning_rate * fraction

        agent_learning_rate = agent_linear_schedule if self.anneal_agent_learning_rate else self.agent_learning_rate
        dual_learning_rate = dual_linear_schedule if self.anneal_dual_learning_rate else self.dual_learning_rate

        self.agent_optimizer = optax.chain(
            optax.clip(self.max_param_update),
            optax.inject_hyperparams(optax.adam)(learning_rate=agent_learning_rate),
        )
        self.dual_optimizer = optax.chain(
            optax.clip(self.max_param_update),
            optax.inject_hyperparams(optax.adam)(learning_rate=dual_learning_rate),
        )

        state = jnp.array([self.env.single_observation_space.sample()])
        action = jnp.array([self.env.single_action_space.sample()])

        agent_params = AgentParams(
            policy_params=self.policy.init(policy_key, state),
            critic_params=self.critic.init(critic_key, state, action)
        )
        
        dual_variable_shape = [np.prod(env.single_action_space.shape).item()]

        # The Lagrange multiplieres
        dual_params = DualParams(
            log_temperature=jnp.full([1], self.init_log_temperature, dtype=jnp.float32),
            log_alpha_mean=jnp.full(dual_variable_shape, self.init_log_alpha_mean, dtype=jnp.float32),
            log_alpha_stddev=jnp.full(dual_variable_shape, self.init_log_alpha_stddev, dtype=jnp.float32),
            log_penalty_temperature=jnp.full([1], self.init_log_temperature, dtype=jnp.float32)
        )

        self.train_state = TrainingState(
            agent_params=agent_params,
            agent_target_params=agent_params,
            dual_params=dual_params,
            agent_optimizer_state=self.agent_optimizer.init(agent_params),
            dual_optimizer_state=self.dual_optimizer.init(dual_params),
            steps=0,
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
    
    def train(self):
        @jax.jit
        def get_action_and_log_prob(policy_params: flax.core.FrozenDict, state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(policy_params, state)
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            log_prob = dist.log_prob(action)
            return action, log_prob, key


        @jax.jit
        def get_log_prob(policy_params: flax.core.FrozenDict, state: np.ndarray, action: np.ndarray):
            dist = self.policy.apply(policy_params, state)
            return dist.log_prob(action)


        @jax.jit
        def update(train_state: TrainingState, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, log_probs: np.ndarray, key: jax.random.PRNGKey):
            def loss_fn(agent_params: flax.core.FrozenDict, dual_params: flax.core.FrozenDict,
                        states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, log_probs: np.ndarray, key: jax.random.PRNGKey
                ):
                # Compute predictions. Atoms for TQC critic loss
                pred_policy = self.policy.apply(agent_params.policy_params, states)
                q_atoms_pred = self.critic.apply(agent_params.critic_params, states[:-1], actions[:-1])
                q_atoms_pred = jnp.transpose(q_atoms_pred, (1, 0, 2)).reshape(self.trace_length - 1, self.nr_total_atoms)
                q_atoms_pred = jnp.expand_dims(q_atoms_pred, axis=2)


                # Compute targets
                target_policy = self.policy.apply(train_state.agent_target_params.policy_params, states)

                a_improvement = target_policy.sample(self.nr_samples, seed=key)

                vmap_critic_call = jax.vmap(self.critic.apply, in_axes=(None, None, 0))
                q_improvement = vmap_critic_call(train_state.agent_target_params.critic_params, states, a_improvement)
                q_improvement = jnp.transpose(q_improvement, (0, 2, 1, 3)).reshape(self.nr_samples, self.trace_length, self.nr_total_atoms)
                q_improvement = jnp.mean(q_improvement, axis=2)

                eval_policy = target_policy

                a_evaluation = eval_policy.sample(self.nr_samples, seed=key)

                value_atoms_target = vmap_critic_call(train_state.agent_target_params.critic_params, states, a_evaluation)
                value_atoms_target = jnp.transpose(value_atoms_target, (0, 2, 1, 3)).reshape(self.nr_samples, self.trace_length, self.nr_total_atoms)
                value_atoms_target = jnp.mean(value_atoms_target, axis=0)
                value_atoms_target = jnp.sort(value_atoms_target)[:, :self.nr_target_atoms]

                ###
                # Retrace calculation defined in rlax and used by acme: https://github.com/deepmind/rlax/blob/master/rlax/_src/multistep.py#L380#L433
                q_t = self.critic.apply(train_state.agent_target_params.critic_params, states[1:-1], actions[1:-1])
                q_t = jnp.transpose(q_t, (1, 0, 2)).reshape(self.trace_length - 2, self.nr_total_atoms)
                q_t = jnp.sort(q_t)[:, :self.nr_target_atoms]
                v_t = value_atoms_target[1:, :]
                r_t = rewards[:-1]
                discount_t = self.gamma * (1 - terminations[:-1])
                log_rhos = target_policy.log_prob(actions) - log_probs
                c_t = self.retrace_lambda * jnp.minimum(1.0, jnp.exp(log_rhos[1:-1]))

                g = r_t[-1] + discount_t[-1] * v_t[-1:, ]

                def _body(acc, xs):
                    reward, discount, c, v, q = xs
                    acc = reward + discount * (v - c * q + c * acc)
                    return acc, acc
                
                _, returns = jax.lax.scan(_body, g, (r_t[:-1], discount_t[:-1], c_t, v_t[:-1, :], q_t), reverse=True)
                returns = jnp.concatenate([returns, g[jnp.newaxis]], axis=0)
                ###

                q_atoms_target = returns


                # Compute policy loss

                # Cast `MultivariateNormalDiag`s to Independent Normals. Allows to satisfy KL constraints per-dimension.
                target_policy = tfd.Independent(tfd.Normal(target_policy.mean(), target_policy.stddev()))
                current_policy = tfd.Independent(tfd.Normal(pred_policy.mean(), pred_policy.stddev()))

                # Convert from log. Use softplus for stability
                temperature = jax.nn.softplus(dual_params.log_temperature) + self.stability_epsilon
                alpha_mean = jax.nn.softplus(dual_params.log_alpha_mean) + self.stability_epsilon
                alpha_stddev = jax.nn.softplus(dual_params.log_alpha_stddev) + self.stability_epsilon
                penalty_temperature = jax.nn.softplus(dual_params.log_penalty_temperature) + self.stability_epsilon

                current_mean = current_policy.distribution.mean()
                current_stddev = current_policy.distribution.stddev()
                target_mean = target_policy.distribution.mean()
                target_stddev = target_policy.distribution.stddev()

                # Optimize the dual function. Equation (9) from the paper. Temperature is the Lagrange multiplier eta.
                tempered_q_values = jax.lax.stop_gradient(q_improvement) / temperature
                normalized_weights = jax.lax.stop_gradient(jax.nn.softmax(tempered_q_values, axis=0))
                q_logsumexp = jax.scipy.special.logsumexp(tempered_q_values, axis=0)
                loss_temperature = temperature * (self.kl_epsilon + jnp.mean(q_logsumexp) - jnp.log(self.nr_samples))
                # Estimate actualized KL
                kl_nonparametric = jnp.sum(normalized_weights * jnp.log(self.nr_samples * normalized_weights + self.stability_epsilon), axis=0)

                # Penalty for violating the action constraint
                diff_out_of_bound = a_improvement - jnp.clip(a_improvement, -1.0, 1.0)
                cost_out_of_bound = -jnp.linalg.norm(diff_out_of_bound, axis=-1)
                tempered_cost = jax.lax.stop_gradient(cost_out_of_bound) / penalty_temperature
                penalty_normalized_weights = jax.lax.stop_gradient(jax.nn.softmax(tempered_cost, axis=0))
                penalty_logsumexp = jax.scipy.special.logsumexp(tempered_cost, axis=0)
                loss_penalty_temperature = penalty_temperature * (self.kl_epsilon_penalty + jnp.mean(penalty_logsumexp) - jnp.log(self.nr_samples))
                # Estimate actualized KL
                penalty_kl_nonparametric = jnp.sum(penalty_normalized_weights * jnp.log(self.nr_samples * penalty_normalized_weights + self.stability_epsilon), axis=0)

                normalized_weights += penalty_normalized_weights
                loss_temperature += loss_penalty_temperature

                # Calculate KL loss for policy with alpha as the Lagrange multiplier

                # Decompose current policy into fixed-mean & fixed-stddev distributions
                # Not part of the original MPO but seems to help with stability
                fixed_stddev_dist = tfd.Independent(tfd.Normal(loc=current_mean, scale=target_stddev))
                fixed_mean_dist = tfd.Independent(tfd.Normal(loc=target_mean, scale=current_stddev))

                loss_policy_mean = jnp.mean(-jnp.sum(fixed_stddev_dist.log_prob(a_improvement) * normalized_weights, axis=0))
                loss_policy_stddev = jnp.mean(-jnp.sum(fixed_mean_dist.log_prob(a_improvement) * normalized_weights, axis=0))

                kl_mean = target_policy.distribution.kl_divergence(fixed_stddev_dist.distribution)
                kl_stddev = target_policy.distribution.kl_divergence(fixed_mean_dist.distribution)

                mean_kl_mean = jnp.mean(kl_mean, axis=0)
                loss_kl_mean = jnp.sum(jax.lax.stop_gradient(alpha_mean) * mean_kl_mean)
                loss_alpha_mean = jnp.sum(alpha_mean * (self.kl_epsilon_mean - jax.lax.stop_gradient(mean_kl_mean)))

                mean_kl_stddev = jnp.mean(kl_stddev, axis=0)
                loss_kl_stddev = jnp.sum(jax.lax.stop_gradient(alpha_stddev) * mean_kl_stddev)
                loss_alpha_stddev = jnp.sum(alpha_stddev * (self.kl_epsilon_stddev - jax.lax.stop_gradient(mean_kl_stddev)))

                # Combine losses
                unconst_policy_loss = loss_policy_mean + loss_policy_stddev
                kl_penalty_loss = loss_kl_mean + loss_kl_stddev
                alpha_loss = loss_alpha_mean + loss_alpha_stddev
                policy_loss = unconst_policy_loss + kl_penalty_loss + alpha_loss + loss_temperature


                # Compute critic loss in TQC style
                cumulative_prob = (jnp.arange(self.nr_total_atoms, dtype=jnp.float32) + 0.5) / self.nr_total_atoms
                cumulative_prob = jnp.expand_dims(cumulative_prob, axis=(0, -1))  # (1, nr_total_atoms, 1)

                delta_i_j = q_atoms_target - q_atoms_pred
                abs_delta_i_j = jnp.abs(delta_i_j)
                huber_loss = jnp.where(abs_delta_i_j <= self.huber_kappa, 0.5 * delta_i_j ** 2, self.huber_kappa * (abs_delta_i_j - 0.5 * self.huber_kappa))
                critic_loss = jnp.mean(jnp.abs(cumulative_prob - (delta_i_j < 0).astype(jnp.float32)) * huber_loss / self.huber_kappa)


                loss = policy_loss + critic_loss


                # Create metrics
                metrics = {
                    "dual/alpha_mean": jnp.mean(alpha_mean),
                    "dual/alpha_stddev": jnp.mean(alpha_stddev),
                    "dual/temperature": jnp.mean(temperature),
                    "dual/penalty_temperature": jnp.mean(penalty_temperature),
                    "loss/unconst_policy_loss": unconst_policy_loss,
                    "loss/temperature_loss": loss_temperature,
                    "loss/critic_loss": critic_loss,
                    "kl/q_kl": jnp.mean(kl_nonparametric) / self.kl_epsilon,
                    "kl/action_penalty_kl": jnp.mean(penalty_kl_nonparametric) / self.kl_epsilon_penalty,
                    "q_vals/q_pred": jnp.mean(q_atoms_pred),
                    "policy/std_dev": jnp.mean(current_stddev)
                }

                # Some metrics seem to cost performance when logged, so they get only logged if necessary
                # TODO: When the jax logger works again, inspect why this is the case
                if self.logging_all_metrics:
                    metrics["loss/policy_mean_loss"] = loss_policy_mean
                    metrics["loss/policy_stddev_loss"] = loss_policy_stddev
                    metrics["loss/kl_penalty_loss"] = kl_penalty_loss
                    metrics["loss/alpha_loss"] = alpha_loss
                    metrics["loss/alpha_mean_loss"] = loss_alpha_mean
                    metrics["loss/alpha_stddev_loss"] = loss_alpha_stddev
                    metrics["kl/policy_mean_kl"] = jnp.mean(mean_kl_mean) / self.kl_epsilon_mean
                    metrics["kl/policy_stddev_kl"] = jnp.mean(mean_kl_stddev) / self.kl_epsilon_stddev
                    metrics["loss/policy_fix_std_logprob"] = jnp.mean(fixed_stddev_dist.log_prob(a_improvement))
                    metrics["loss/policy_fix_mean_logprob"] = jnp.mean(fixed_mean_dist.log_prob(a_improvement))
                    metrics["loss/a_improvement"] = jnp.mean(a_improvement)
                    metrics["loss/target_mean"] = jnp.mean(target_mean)
                    metrics["loss/target_stddev"] = jnp.mean(target_stddev)
                    metrics["Q_vals/Q_target"] = jnp.mean(q_atoms_target)
                    metrics["Q_vals/Q_improvement"] = jnp.mean(q_improvement)
                    metrics["policy/mean"] = jnp.mean(current_mean)

                return loss, (metrics)


            keys = jax.random.split(key, self.batch_size + 1)
            key, keys = keys[0], keys[1:]

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            (loss, (metrics)), (agent_gradients, dual_gradients) = grad_loss_fn(train_state.agent_params, train_state.dual_params, states, actions, rewards, terminations, log_probs, keys)

            agent_gradients_norm = optax.global_norm(agent_gradients)
            dual_gradients_norm = optax.global_norm(dual_gradients)

            agent_updates, agent_optimizer_state = self.agent_optimizer.update(agent_gradients, train_state.agent_optimizer_state, train_state.agent_params)
            dual_updates, dual_optimizer_state = self.dual_optimizer.update(dual_gradients, train_state.dual_optimizer_state, train_state.dual_params)

            agent_params = optax.apply_updates(train_state.agent_params, agent_updates)
            steps = train_state.steps + 1
            agent_target_params = optax.periodic_update(agent_params, train_state.agent_target_params, steps, self.update_period)

            dual_params = optax.apply_updates(train_state.dual_params, dual_updates)
            dual_params = DualParams(
                log_temperature=jnp.maximum(dual_params.log_temperature, self.min_log_temperature),
                log_alpha_mean=jnp.maximum(dual_params.log_alpha_mean, self.min_log_alpha),
                log_alpha_stddev=jnp.maximum(dual_params.log_alpha_stddev, self.min_log_alpha),
                log_penalty_temperature=jnp.maximum(dual_params.log_penalty_temperature, self.min_log_temperature)
            )

            new_train_state = TrainingState(
                agent_params=agent_params,
                agent_target_params=agent_target_params,
                dual_params=dual_params,
                agent_optimizer_state=agent_optimizer_state,
                dual_optimizer_state=dual_optimizer_state,
                steps=steps
            )

            metrics["lr/agent_learning_rate"] = new_train_state.agent_optimizer_state[1].hyperparams["learning_rate"]
            metrics["lr/dual_learning_rate"] = new_train_state.dual_optimizer_state[1].hyperparams["learning_rate"]
            metrics["gradients/agent_gradients_norm"] = agent_gradients_norm
            metrics["gradients/dual_gradients_norm"] = dual_gradients_norm

            return new_train_state, metrics, key
        

        @jax.jit
        def get_deterministic_action(policy_params: flax.core.FrozenDict, state: np.ndarray):
            dist = self.policy.apply(policy_params, state)
            action = dist.mode()
            return self.get_processed_action(action)


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.trace_length, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng)

        state_stack = np.zeros((self.nr_envs, self.trace_length) + self.env.single_observation_space.shape, dtype=np.float32)
        action_stack = np.zeros((self.nr_envs, self.trace_length) + self.env.single_action_space.shape, dtype=np.float32)
        reward_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)
        terminated_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)
        log_prob_stack = np.zeros((self.nr_envs, self.trace_length), dtype=np.float32)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
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
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
                log_prob = get_log_prob(self.train_state.agent_target_params.policy_params, state, action)
            else:
                action, log_prob, self.key = get_action_and_log_prob(self.train_state.agent_target_params.policy_params, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
            done = terminated | truncated
            for i, single_done in enumerate(done):
                if single_done:
                    saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            nr_episodes += dones_this_rollout
            global_step += self.nr_envs
            
            state_stack = np.roll(state_stack, shift=-1, axis=1)
            action_stack = np.roll(action_stack, shift=-1, axis=1)
            reward_stack = np.roll(reward_stack, shift=-1, axis=1)
            terminated_stack = np.roll(terminated_stack, shift=-1, axis=1)
            log_prob_stack = np.roll(log_prob_stack, shift=-1, axis=1)

            state_stack[:, -1] = state
            action_stack[:, -1] = action
            reward_stack[:, -1] = reward
            terminated_stack[:, -1] = terminated
            log_prob_stack[:, -1] = log_prob

            if global_step / self.nr_envs >= self.trace_length:
                replay_buffer.add(state_stack, action_stack, reward_stack, terminated_stack, log_prob_stack)

            state = next_state

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_actions, batch_rewards, batch_terminations, batch_log_probs = replay_buffer.sample(self.batch_size)


            # Optimizing - Critic, policy, Lagrange multipliers
            if should_optimize:
                self.train_state, optimization_metrics, self.key = update(
                    self.train_state, batch_states, batch_actions, batch_rewards, batch_terminations, batch_log_probs, self.key)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                while True:
                    processed_action = get_deterministic_action(self.train_state.agent_target_params.policy_params, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                state, _ = self.env.reset()
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
            "train_state": self.train_state
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


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = MPO(config, env, run_path, writer)

        target = {
            "train_state": model.train_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.train_state = checkpoint["train_state"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_params: flax.core.FrozenDict, state: np.ndarray):
            dist = self.policy.apply(policy_params, state)
            action = dist.mode()
            return self.get_processed_action(action)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                processed_action = get_action(self.train_state.agent_params.policy_params, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
            return GeneralProperties
