import os
import shutil
import json
import math
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

from rl_x.algorithms.flashsac.flax.general_properties import GeneralProperties
from rl_x.algorithms.flashsac.flax.policy import get_policy
from rl_x.algorithms.flashsac.flax.critic import get_critic
from rl_x.algorithms.flashsac.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.flashsac.flax.layers import project_params
from rl_x.algorithms.flashsac.flax.rl_train_state import RLTrainState
from rl_x.algorithms.flashsac.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.flashsac.flax import reward_normalizer
from rl_x.algorithms.flashsac.flax import noise_repeat

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
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
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
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.os_shape = self.train_env.single_observation_space.shape
        self.action_dim = self.train_env.single_action_space.shape[0]

        target_sigma = config.algorithm.target_entropy_sigma
        self.target_entropy = 0.5 * self.action_dim * math.log(2.0 * math.pi * math.e * target_sigma ** 2)
        if self.logging_frequency % self.nr_envs != 0:
            raise ValueError("The logging frequency must be a multiple of the number of environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key, noise_key = jax.random.split(self.key, 5)

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)
        self.entropy_coefficient = EntropyCoefficient(initial_value=config.algorithm.init_entropy_coefficient)

        total_updates = max(1, (self.total_timesteps // self.nr_envs) * self.updates_per_step)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.learning_rate_init,
            peak_value=self.learning_rate_peak,
            warmup_steps=self.learning_rate_warmup_steps,
            decay_steps=total_updates,
            end_value=self.learning_rate_end,
        )

        dummy_obs = jnp.zeros((1,) + self.os_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dim), dtype=jnp.float32)
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
        self.noise_state = noise_repeat.init_noise_state(self.nr_envs, self.action_dim, noise_key)
        self.reward_normalizer_state = reward_normalizer.init_reward_normalizer_state(self.nr_envs)
        self.update_step_count = 0

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        @jax.jit
        def act_and_step_noise(policy_state, state, noise_state, key):
            mean, std = self.policy.apply(
                {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                state, train=False,
            )
            noise_state = noise_repeat.step_noise(noise_state, key, self.zeta_cdf)
            action = self.policy.sample_with_noise(mean, std, noise_state["noise"], 1.0)
            return action, noise_state


        @jax.jit
        def deterministic_act(policy_state, state):
            mean, _ = self.policy.apply(
                {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                state, train=False,
            )
            action = self.policy.deterministic_action(mean)
            return self.get_processed_action(action)


        @jax.jit
        def update_reward_normalizer(reward_normalizer_state, reward, terminated, truncated):
            return reward_normalizer.update_reward_normalizer(reward_normalizer_state, reward, terminated, truncated, self.gamma)


        @jax.jit
        def update_step(policy_state, critic_state, target_critic_state, entropy_coefficient_state, reward_normalizer_state, states, next_states, actions, rewards, dones, truncations, effective_n_steps, do_policy, key):
            policy_key, critic_key = jax.random.split(key)

            if self.normalize_reward:
                rewards = reward_normalizer.normalize_reward(reward_normalizer_state, rewards, self.normalized_g_max)

            def policy_loss_fn(policy_params, batch_stats):
                all_observations = jnp.concatenate([states, next_states], axis=0)
                (all_means, all_stds), policy_state_update = self.policy.apply(
                    {"params": policy_params, "batch_stats": batch_stats},
                    all_observations, train=True, mutable=["batch_stats"],
                )
                mean = all_means[:states.shape[0]]
                std = all_stds[:states.shape[0]]
                action, log_prob = self.policy.sample_and_log_prob(mean, std, policy_key)
                (q_values, _), _ = self.critic.apply(
                    {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                    states, action, train=False, mutable=["batch_stats"],
                )
                q = jnp.minimum(q_values[0], q_values[1])
                alpha = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_state.params))
                loss = jnp.mean(alpha * log_prob - q)
                entropy = -jnp.mean(log_prob)
                return loss, (policy_state_update, entropy, jnp.mean(q))

            def entropy_loss_fn(entropy_coefficient_params, entropy):
                alpha = self.entropy_coefficient.apply(entropy_coefficient_params)
                loss = alpha * (entropy - self.target_entropy)
                return loss, alpha

            def apply_policy_update(carry):
                policy_state, entropy_coefficient_state = carry
                (policy_loss, (policy_state_update, entropy, policy_q_mean)), policy_gradients = jax.value_and_grad(
                    policy_loss_fn, has_aux=True,
                )(policy_state.params, policy_state.batch_stats)
                policy_state = policy_state.apply_gradients(grads=policy_gradients)
                policy_state = policy_state.replace(
                    params=project_params(policy_state.params),
                    batch_stats=policy_state_update["batch_stats"],
                )
                (entropy_loss, alpha), entropy_coefficient_gradients = jax.value_and_grad(
                    entropy_loss_fn, has_aux=True,
                )(entropy_coefficient_state.params, stop_gradient(entropy))
                entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_coefficient_gradients)
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
                do_policy, apply_policy_update, skip_policy_update, (policy_state, entropy_coefficient_state),
            )

            def critic_loss_fn(critic_params, batch_stats):
                next_mean, next_std = self.policy.apply(
                    {"params": policy_state.params, "batch_stats": policy_state.batch_stats},
                    next_states, train=False,
                )
                next_action, next_log_prob = self.policy.sample_and_log_prob(next_mean, next_std, critic_key)
                alpha = stop_gradient(self.entropy_coefficient.apply(entropy_coefficient_state.params))

                all_observations = jnp.concatenate([states, next_states], axis=0)
                all_actions = jnp.concatenate([actions, next_action], axis=0)

                (_, target_log_probabilities), target_critic_state_update = self.critic.apply(
                    {"params": target_critic_state.params, "batch_stats": target_critic_state.batch_stats},
                    all_observations, all_actions, train=True, mutable=["batch_stats"],
                )
                next_log_probabilities = target_log_probabilities[:, states.shape[0]:, :]
                next_values = jnp.sum(
                    jnp.exp(next_log_probabilities) * jnp.linspace(self.v_min, self.v_max, self.nr_atoms), axis=-1,
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
                    all_observations, all_actions, train=True, mutable=["batch_stats"],
                )
                predicted_log_probabilities = predicted_log_probabilities[:, :states.shape[0], :]
                cross_entropy = -jnp.sum(
                    target_probabilities[None, :, :] * predicted_log_probabilities, axis=-1,
                )
                loss = jnp.mean(cross_entropy)
                return loss, (critic_state_update, target_critic_state_update, jnp.mean(next_values))

            (critic_loss, (critic_state_update, target_critic_state_update, target_q_mean)), critic_gradients = jax.value_and_grad(
                critic_loss_fn, has_aux=True,
            )(critic_state.params, critic_state.batch_stats)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)
            critic_state = critic_state.replace(params=project_params(critic_state.params), batch_stats=critic_state_update["batch_stats"])

            new_target_params = jax.tree_util.tree_map(lambda p, tp: self.tau * p + (1 - self.tau) * tp, critic_state.params, target_critic_state.params)
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
            return policy_state, critic_state, target_critic_state, entropy_coefficient_state, metrics


        self.set_train_mode()

        replay_buffer = ReplayBuffer(self.buffer_size, self.nr_envs, self.os_shape, self.action_dim, self.n_steps, self.gamma)
        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        rng = np.random.default_rng(self.seed)
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
            if global_step < self.learning_starts:
                action = rng.uniform(-1.0, 1.0, size=(self.nr_envs, self.action_dim)).astype(np.float32)
            else:
                self.key, action_key = jax.random.split(self.key)
                action, self.noise_state = act_and_step_noise(
                    self.policy_state, jnp.asarray(state, dtype=jnp.float32), self.noise_state, action_key,
                )
                action = jax.device_get(action)
            processed_action = self.get_processed_action(action)
            next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            dones_this_rollout = 0
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                    saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for k, info_value in self.train_env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(k, []).extend(np.asarray(info_value).reshape(-1).tolist())
            nr_episodes += dones_this_rollout

            if self.normalize_reward:
                self.reward_normalizer_state = update_reward_normalizer(
                    self.reward_normalizer_state,
                    jnp.asarray(reward, dtype=jnp.float32),
                    jnp.asarray(terminated, dtype=jnp.bool_),
                    jnp.asarray(truncated, dtype=jnp.bool_),
                )

            replay_buffer.add(state, actual_next_state, action, reward, done.astype(np.float32), truncated.astype(np.float32))
            state = next_state
            global_step += self.nr_envs
            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_optimize = global_step >= self.learning_starts and replay_buffer.can_sample()
            should_evaluate = self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0
            should_try_to_save = should_optimize and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing
            if should_optimize:
                for _ in range(self.updates_per_step):
                    states_b, next_states_b, actions_b, rewards_b, dones_b, truncs_b, eff_n_b = replay_buffer.sample(rng, self.batch_size)
                    self.key, upd_key = jax.random.split(self.key)
                    do_policy = jnp.array(self.update_step_count % self.policy_delay == 0)
                    self.policy_state, self.critic_state, self.target_critic_state, self.entropy_coefficient_state, optimization_metrics = update_step(
                        self.policy_state, self.critic_state, self.target_critic_state, self.entropy_coefficient_state, self.reward_normalizer_state,
                        jnp.asarray(states_b), jnp.asarray(next_states_b), jnp.asarray(actions_b),
                        jnp.asarray(rewards_b), jnp.asarray(dones_b), jnp.asarray(truncs_b), jnp.asarray(eff_n_b),
                        do_policy, upd_key,
                    )
                    for key, value in optimization_metrics.items():
                        optimization_metrics_collection.setdefault(key, []).append(value)
                    self.update_step_count += 1
                    nr_updates += 1

            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                eval_episode_returns = np.zeros(self.nr_envs, dtype=np.float64)
                eval_episode_lengths = np.zeros(self.nr_envs, dtype=np.int64)
                while True:
                    eval_action = deterministic_act(self.policy_state, jnp.asarray(eval_state, dtype=jnp.float32))
                    eval_state, eval_reward, eval_term, eval_trunc, _ = self.eval_env.step(jax.device_get(eval_action))
                    eval_done = eval_term | eval_trunc
                    eval_episode_returns += np.asarray(eval_reward)
                    eval_episode_lengths += 1
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(eval_episode_returns[i])
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(eval_episode_lengths[i])
                            eval_episode_returns[i] = 0.0
                            eval_episode_lengths[i] = 0
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
                optimization_metrics = {
                    key: float(np.nanmean(value))
                    for key, value in optimization_metrics_collection.items()
                    if not np.isnan(np.asarray(value)).all()
                }
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
            "critic": self.critic_state,
            "target_critic": self.target_critic_state,
            "entropy_coefficient": self.entropy_coefficient_state,
            "reward_normalizer": self.reward_normalizer_state,
            "update_step_count": np.asarray(self.update_step_count, dtype=np.int64),
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
        model = FlashSAC(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "target_critic": model.target_critic_state,
            "entropy_coefficient": model.entropy_coefficient_state,
            "reward_normalizer": model.reward_normalizer_state,
            "update_step_count": np.asarray(0, dtype=np.int64),
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.target_critic_state = checkpoint["target_critic"]
        model.entropy_coefficient_state = checkpoint["entropy_coefficient"]
        model.reward_normalizer_state = checkpoint["reward_normalizer"]
        model.update_step_count = int(checkpoint["update_step_count"])
        shutil.rmtree(checkpoint_dir)
        return model


    def test(self, episodes):
        @jax.jit
        def act(state):
            mean, _ = self.policy.apply(
                {"params": self.policy_state.params, "batch_stats": self.policy_state.batch_stats},
                state, train=False,
            )
            return self.get_processed_action(self.policy.deterministic_action(mean))

        self.set_eval_mode()
        state, _ = self.eval_env.reset()
        episode_returns = np.zeros(self.nr_envs, dtype=np.float64)
        completed_episodes = 0
        while completed_episodes < episodes:
            action = act(jnp.asarray(state, dtype=jnp.float32))
            state, reward, terminated, truncated, _ = self.eval_env.step(jax.device_get(action))
            episode_returns += np.asarray(reward)
            for index, single_done in enumerate(terminated | truncated):
                if single_done:
                    completed_episodes += 1
                    rlx_logger.info(f"Episode {completed_episodes} - Return: {episode_returns[index]}")
                    episode_returns[index] = 0.0
                    if completed_episodes == episodes:
                        break


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
