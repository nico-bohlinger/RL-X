import os
import shutil
import json
import logging
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.bro.flax.general_properties import GeneralProperties
from rl_x.algorithms.bro.flax.policy import get_policy
from rl_x.algorithms.bro.flax.critic import get_critic
from rl_x.algorithms.bro.flax.entropy_coefficient import EntropyCoefficient, Adjustment, calculate_init_log_param
from rl_x.algorithms.bro.flax.replay_buffer import ReplayBuffer

rlx_logger = logging.getLogger("rl_x")

LOG_VAL_MIN = -10.0
LOG_VAL_MAX = 7.5


def quantile_huber_loss(td, taus, kappa=1.0):
    huber = jnp.where(jnp.abs(td) <= kappa, 0.5 * td ** 2, kappa * (jnp.abs(td) - 0.5 * kappa))
    mask = stop_gradient(jnp.where(td < 0, 1.0, 0.0))
    return (jnp.abs(taus[..., None] - mask) * huber / kappa).sum(axis=1).mean()


class BRO:
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
        self.policy_learning_rate = config.algorithm.policy_learning_rate
        self.critic_learning_rate = config.algorithm.critic_learning_rate
        self.entropy_coefficient_learning_rate = config.algorithm.entropy_coefficient_learning_rate
        self.adjustment_learning_rate = config.algorithm.adjustment_learning_rate
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.updates_per_step = config.algorithm.updates_per_step
        self.gamma = config.algorithm.gamma
        self.tau = config.algorithm.tau
        self.distributional = config.algorithm.distributional
        self.nr_quantiles = config.algorithm.nr_quantiles
        self.pessimism = config.algorithm.pessimism
        self.kl_target = config.algorithm.kl_target
        self.std_multiplier = config.algorithm.std_multiplier
        self.use_optimistic_exploration = config.algorithm.use_optimistic_exploration
        self.first_reset_step = config.algorithm.first_reset_step
        self.reset_interval = config.algorithm.reset_interval
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dim = self.as_shape[0]
        self.env_as_low = self.train_env.single_action_space.low
        self.env_as_high = self.train_env.single_action_space.high
        if config.algorithm.target_entropy == "auto":
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(config.algorithm.target_entropy)

        taus = jnp.arange(0, self.nr_quantiles + 1) / self.nr_quantiles
        self.quantile_taus = ((taus[1:] + taus[:-1]) / 2.0)[None, ...]

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.rng = np.random.default_rng(self.seed)
        self.build_states(jax.random.PRNGKey(self.seed))

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def build_states(self, master_key):
        self.policy, self.optimistic_policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)
        self.entropy_coefficient = EntropyCoefficient(initial_value=self.config.algorithm.init_entropy_coefficient)

        init_optimism_raw = calculate_init_log_param(self.config.algorithm.init_optimism, LOG_VAL_MIN, LOG_VAL_MAX)
        init_regularizer_raw = calculate_init_log_param(self.config.algorithm.init_regularizer, LOG_VAL_MIN, LOG_VAL_MAX)
        self.optimism = Adjustment(init_value=init_optimism_raw, log_val_min=LOG_VAL_MIN, log_val_max=LOG_VAL_MAX)
        self.regularizer = Adjustment(init_value=init_regularizer_raw, log_val_min=LOG_VAL_MIN, log_val_max=LOG_VAL_MAX)

        self.key, policy_key, optimistic_policy_key, critic_key, entropy_coefficient_key, optimism_key, regularizer_key = jax.random.split(master_key, 7)

        dummy_obs = jnp.zeros((1,) + self.os_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_obs),
            tx=optax.adamw(learning_rate=self.policy_learning_rate),
        )
        self.optimistic_policy_state = TrainState.create(
            apply_fn=self.optimistic_policy.apply,
            params=self.optimistic_policy.init(optimistic_policy_key, dummy_obs, dummy_action, dummy_action, self.std_multiplier),
            tx=optax.adamw(learning_rate=self.policy_learning_rate),
        )
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adamw(learning_rate=self.critic_learning_rate),
        )
        self.target_critic_params = critic_params
        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.adam(learning_rate=self.entropy_coefficient_learning_rate, b1=0.5),
        )
        self.optimism_state = TrainState.create(
            apply_fn=self.optimism.apply,
            params=self.optimism.init(optimism_key),
            tx=optax.adam(learning_rate=self.adjustment_learning_rate, b1=0.5),
        )
        self.regularizer_state = TrainState.create(
            apply_fn=self.regularizer.apply,
            params=self.regularizer.init(regularizer_key),
            tx=optax.adam(learning_rate=self.adjustment_learning_rate, b1=0.5),
        )


    def train(self):
        @jax.jit
        def get_action_pessimistic(policy_state, state, key):
            mean, std = self.policy.apply(policy_state.params, state)
            key, subkey = jax.random.split(key)
            return self.policy.sample_action(mean, std, subkey), key


        @jax.jit
        def get_action_optimistic(policy_state, optimistic_policy_state, state, key):
            pessimistic_mean, pessimistic_std = self.policy.apply(policy_state.params, state)
            optimistic_mean, optimistic_std = self.optimistic_policy.apply(
                optimistic_policy_state.params, state, pessimistic_mean, pessimistic_std, self.std_multiplier,
            )
            key, subkey = jax.random.split(key)
            return self.optimistic_policy.sample_action(optimistic_mean, optimistic_std, subkey), key


        @jax.jit
        def get_deterministic_action(policy_state, state):
            mean, _ = self.policy.apply(policy_state.params, state)
            action = self.policy.deterministic_action(mean)
            return self.get_processed_action(action)


        @jax.jit
        def update_step(policy_state, optimistic_policy_state, critic_state, target_critic_params, entropy_coefficient_state, optimism_state, regularizer_state, states, next_states, actions, rewards, terms, key):
            critic_key, policy_key, optimistic_policy_key = jax.random.split(key, 3)

            next_mean, next_std = self.policy.apply(policy_state.params, next_states)
            next_action, next_log_prob = self.policy.sample_and_log_prob(next_mean, next_std, critic_key)
            entropy_coefficient = self.entropy_coefficient.apply(entropy_coefficient_state.params)
            next_q1, next_q2 = self.critic.apply(target_critic_params, next_states, next_action)

            if self.distributional:
                next_q = (next_q1 + next_q2) / 2.0 - self.pessimism * jnp.abs(next_q1 - next_q2) / 2.0
                target_q = rewards[:, None, None] + self.gamma * (1.0 - terms[:, None, None]) * next_q[:, None, :]
                target_q = target_q - self.gamma * entropy_coefficient * (1.0 - terms[:, None, None]) * next_log_prob[:, None, None]
                target_q = stop_gradient(target_q)

                def critic_loss_fn(critic_params):
                    pred_q1, pred_q2 = self.critic.apply(critic_params, states, actions)
                    td1 = target_q - pred_q1[..., None]
                    td2 = target_q - pred_q2[..., None]
                    return quantile_huber_loss(td1, self.quantile_taus) + quantile_huber_loss(td2, self.quantile_taus), (pred_q1.mean(), pred_q2.mean())

                (critic_loss_value, (q1_mean, q2_mean)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
            else:
                next_q = (next_q1 + next_q2) / 2.0 - self.pessimism * jnp.abs(next_q1 - next_q2) / 2.0
                target_q = rewards + self.gamma * (1.0 - terms) * (next_q - entropy_coefficient * next_log_prob)
                target_q = stop_gradient(target_q)

                def critic_loss_fn(critic_params):
                    pred_q1, pred_q2 = self.critic.apply(critic_params, states, actions)
                    return ((pred_q1 - target_q) ** 2 + (pred_q2 - target_q) ** 2).mean(), (pred_q1.mean(), pred_q2.mean())

                (critic_loss_value, (q1_mean, q2_mean)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)

            critic_state = critic_state.apply_gradients(grads=critic_grads)
            new_target_params = jax.tree_util.tree_map(
                lambda p, tp: self.tau * p + (1.0 - self.tau) * tp,
                critic_state.params, target_critic_params,
            )

            def policy_loss_fn(policy_params):
                mean, std = self.policy.apply(policy_params, states)
                action, log_prob = self.policy.sample_and_log_prob(mean, std, policy_key)
                q1, q2 = self.critic.apply(critic_state.params, states, action)
                q = (q1 + q2) / 2.0 - self.pessimism * jnp.abs(q1 - q2) / 2.0
                if self.distributional:
                    q = q.mean(axis=-1)
                coefficient = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_state.params))
                loss = jnp.mean(coefficient * log_prob - q)
                entropy = -jnp.mean(log_prob)
                return loss, (entropy, jnp.mean(q))

            (policy_loss_value, (entropy, policy_q_mean)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_state.params)
            policy_state = policy_state.apply_gradients(grads=policy_grads)

            pessimistic_mean, pessimistic_std = self.policy.apply(policy_state.params, states)
            pessimistic_mean = stop_gradient(pessimistic_mean)
            pessimistic_std = stop_gradient(pessimistic_std)

            def optimistic_policy_loss_fn(optimistic_policy_params):
                optimistic_mean, optimistic_std = self.optimistic_policy.apply(
                    optimistic_policy_params, states, pessimistic_mean, pessimistic_std, self.std_multiplier,
                )
                action, _ = self.optimistic_policy.sample_and_log_prob(optimistic_mean, optimistic_std, optimistic_policy_key)
                q1, q2 = self.critic.apply(critic_state.params, states, action)
                optimism = stop_gradient(self.optimism.apply(optimism_state.params))
                regularizer = stop_gradient(self.regularizer.apply(regularizer_state.params))
                q_ub = (q1 + q2) / 2.0 + optimism * jnp.abs(q1 - q2) / 2.0
                if self.distributional:
                    q_ub = q_ub.mean(axis=-1)
                effective_optimistic_std = optimistic_std / self.std_multiplier
                kl = (jnp.log(pessimistic_std / effective_optimistic_std) + (effective_optimistic_std ** 2 + (optimistic_mean - pessimistic_mean) ** 2) / (2.0 * pessimistic_std ** 2) - 0.5).sum(axis=-1)
                loss = (-q_ub).mean() + regularizer * kl.mean()
                return loss, kl.mean()

            (optimistic_policy_loss_value, kl_mean), optimistic_policy_grads = jax.value_and_grad(optimistic_policy_loss_fn, has_aux=True)(optimistic_policy_state.params)
            optimistic_policy_state = optimistic_policy_state.apply_gradients(grads=optimistic_policy_grads)

            def entropy_coefficient_loss_fn(entropy_coefficient_params):
                coefficient = self.entropy_coefficient.apply(entropy_coefficient_params)
                return jnp.mean(coefficient * (stop_gradient(entropy) - self.target_entropy)), coefficient

            (entropy_coefficient_loss_value, entropy_coefficient_value), entropy_coefficient_grads = jax.value_and_grad(entropy_coefficient_loss_fn, has_aux=True)(entropy_coefficient_state.params)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_coefficient_grads)

            empirical_kl = kl_mean / self.action_dim

            def optimism_loss_fn(params):
                optimism = self.optimism.apply(params)
                return (optimism - self.pessimism) * (stop_gradient(empirical_kl) - self.kl_target), optimism

            (_, optimism_value), optimism_grads = jax.value_and_grad(optimism_loss_fn, has_aux=True)(optimism_state.params)
            optimism_state = optimism_state.apply_gradients(grads=optimism_grads)

            def regularizer_loss_fn(params):
                regularizer = self.regularizer.apply(params)
                return -regularizer * (stop_gradient(empirical_kl) - self.kl_target), regularizer

            (_, regularizer_value), regularizer_grads = jax.value_and_grad(regularizer_loss_fn, has_aux=True)(regularizer_state.params)
            regularizer_state = regularizer_state.apply_gradients(grads=regularizer_grads)

            metrics = {
                "loss/critic_loss": critic_loss_value,
                "loss/policy_loss": policy_loss_value,
                "loss/optimistic_policy_loss": optimistic_policy_loss_value,
                "loss/entropy_coefficient_loss": entropy_coefficient_loss_value,
                "entropy/entropy": entropy,
                "entropy/entropy_coefficient": entropy_coefficient_value,
                "optimism/value": optimism_value,
                "regularizer/value": regularizer_value,
                "kl/empirical_kl": empirical_kl,
                "q/q1_mean": q1_mean,
                "q/q2_mean": q2_mean,
                "q/policy_q_mean": policy_q_mean,
            }
            return policy_state, optimistic_policy_state, critic_state, new_target_params, entropy_coefficient_state, optimism_state, regularizer_state, metrics


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.os_shape, self.as_shape, self.rng)
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        reset_steps = set()
        if self.reset_interval > 0:
            reset_steps.add(self.first_reset_step)
            reset_step = self.reset_interval
            while reset_step < self.total_timesteps:
                reset_steps.add(reset_step)
                reset_step += self.reset_interval
        reset_steps = deque(sorted(reset_steps))

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
                processed_action = np.stack([self.train_env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                if self.use_optimistic_exploration:
                    action, self.key = get_action_optimistic(self.policy_state, self.optimistic_policy_state, jnp.asarray(state, dtype=jnp.float32), self.key)
                else:
                    action, self.key = get_action_pessimistic(self.policy_state, jnp.asarray(state, dtype=jnp.float32), self.key)
                processed_action = self.get_processed_action(action)

            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for k, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(k, []).extend(info_value)
            replay_buffer.add(state, actual_next_state, action, reward, terminated)
            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout
            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_reset = bool(reset_steps) and global_step >= reset_steps[0]
            should_optimize = should_learning_start
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Resetting
            if should_reset:
                rlx_logger.info(f"Resetting all networks at step {global_step}")
                while reset_steps and global_step >= reset_steps[0]:
                    reset_steps.popleft()
                self.build_states(jax.random.PRNGKey(self.seed))


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)
                    self.key, update_key = jax.random.split(self.key)
                    (self.policy_state, self.optimistic_policy_state, self.critic_state, self.target_critic_params,
                     self.entropy_coefficient_state, self.optimism_state, self.regularizer_state, optimization_metrics) = update_step(
                        self.policy_state, self.optimistic_policy_state, self.critic_state, self.target_critic_params,
                        self.entropy_coefficient_state, self.optimism_state, self.regularizer_state,
                        jnp.asarray(batch_states), jnp.asarray(batch_next_states), jnp.asarray(batch_actions),
                        jnp.asarray(batch_rewards), jnp.asarray(batch_terminations), update_key,
                    )
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                while True:
                    eval_action = get_deterministic_action(self.policy_state, jnp.asarray(eval_state, dtype=jnp.float32))
                    eval_state, _, eval_term, eval_trunc, eval_info = self.eval_env.step(jax.device_get(eval_action))
                    eval_done = eval_term | eval_trunc
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
                mean_return = float(np.mean(saving_return_buffer))
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
                    for info_name, info_vals in step_info_collection.items():
                        metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                        metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                        mean_value = np.mean(info_vals)
                        if mean_value == mean_value:
                            metric_dict[f"{metric_group}/{info_name}"] = mean_value
                time_metrics = {k: np.mean(v) for k, v in time_metrics_collection.items()}
                optimization_metrics = {k: float(np.mean(v)) for k, v in optimization_metrics_collection.items()}
                evaluation_metrics = {k: float(np.mean(v)) for k, v in evaluation_metrics_collection.items()}
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


    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "optimistic_policy": self.optimistic_policy_state,
            "critic": self.critic_state,
            "target_critic_params": self.target_critic_params,
            "entropy_coefficient": self.entropy_coefficient_state,
            "optimism": self.optimism_state,
            "regularizer": self.regularizer_state,
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
        model = BRO(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "optimistic_policy": model.optimistic_policy_state,
            "critic": model.critic_state,
            "target_critic_params": model.target_critic_params,
            "entropy_coefficient": model.entropy_coefficient_state,
            "optimism": model.optimism_state,
            "regularizer": model.regularizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.optimistic_policy_state = checkpoint["optimistic_policy"]
        model.critic_state = checkpoint["critic"]
        model.target_critic_params = checkpoint["target_critic_params"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.optimism_state = checkpoint["optimism"]
        model.regularizer_state = checkpoint["regularizer"]
        shutil.rmtree(checkpoint_dir)
        return model


    def test(self, episodes):
        @jax.jit
        def act(state):
            mean, _ = self.policy.apply(self.policy_state.params, state)
            action = self.policy.deterministic_action(mean)
            return self.get_processed_action(action)

        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                action = act(jnp.asarray(state, dtype=jnp.float32))
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
