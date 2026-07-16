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
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.reppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.reppo.flax.policy import get_policy
from rl_x.algorithms.reppo.flax.critic import get_critic
from rl_x.algorithms.reppo.flax.batch import Batch
from rl_x.algorithms.reppo.flax import observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class REPPO:
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
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.nr_minibatches = config.algorithm.nr_minibatches
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.nr_bins = config.algorithm.nr_bins
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.kl_bound = config.algorithm.kl_bound
        self.auxiliary_loss_coefficient = config.algorithm.auxiliary_loss_coefficient
        self.nr_kl_samples = config.algorithm.nr_kl_samples
        self.normalize_observation = config.algorithm.normalize_observation
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.minibatch_size = self.batch_size // self.nr_minibatches
        self.action_dimension = self.train_env.single_action_space.shape[0]
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = (self.action_dimension,)
        self.target_entropy = self.action_dimension * config.algorithm.target_entropy_multiplier
        bin_width = (self.v_max - self.v_min) / (self.nr_bins - 1)
        self.hl_gauss_centers = jnp.linspace(self.v_min, self.v_max, self.nr_bins, dtype=jnp.float32)
        self.hl_gauss_support = jnp.linspace(self.v_min - bin_width / 2, self.v_max + bin_width / 2, self.nr_bins + 1, dtype=jnp.float32)
        self.hl_gauss_sigma = bin_width * 0.75

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.batch_size % self.nr_minibatches != 0:
            raise ValueError("The rollout batch size must be divisible by the number of minibatches.")
        if self.evaluation_frequency != -1 and self.evaluation_frequency % self.batch_size != 0:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.policy, self.get_processed_action = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        dummy_observation = jnp.zeros((1,) + self.os_shape, dtype=jnp.float32)
        dummy_action = jnp.zeros((1, self.action_dimension), dtype=jnp.float32)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            ),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dummy_observation, dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            ),
        )
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def train(self):
        @jax.jit
        def update_observation_normalizer(observation_normalizer_state, observations):
            return observation_normalizer.update_observation_normalizer(observation_normalizer_state, observations)


        @jax.jit
        def preprocess_observation(observation_normalizer_state, observation):
            if self.normalize_observation:
                return observation_normalizer.normalize_observation(observation_normalizer_state, observation)
            return observation


        @jax.jit
        def get_action(policy_state, state, key):
            loc, log_std, _, _ = self.policy.apply(policy_state.params, state)
            key, subkey = jax.random.split(key)
            action, _ = self.policy.sample_and_log_prob(loc, log_std, subkey)
            return action, key


        @jax.jit
        def evaluate_next(policy_state, critic_state, next_state, reward, key):
            key, subkey = jax.random.split(key)
            loc, log_std, log_entropy_coefficient, _ = self.policy.apply(policy_state.params, next_state)
            next_action, next_log_prob = self.policy.sample_and_log_prob(loc, log_std, subkey)
            next_features, next_logits, _, _ = self.critic.apply(critic_state.params, next_state, next_action)
            next_value = jnp.sum(jax.nn.softmax(next_logits, axis=-1) * self.hl_gauss_centers, axis=-1)
            entropy_coefficient = jnp.exp(log_entropy_coefficient).squeeze()
            soft_reward = reward - self.gamma * next_log_prob * entropy_coefficient
            return next_features, next_value, soft_reward, key


        @jax.jit
        def get_deterministic_action(policy_state, state):
            loc, _, _, _ = self.policy.apply(policy_state.params, state)
            return self.policy.deterministic_action(loc)


        @jax.jit
        def compute_td_lambda_targets(soft_rewards, next_values, terminations, truncations):
            def step(carry, t):
                lambda_return = carry
                value_t = next_values[t]
                done_t = terminations[t]
                truncated_t = truncations[t]
                lambda_sum = self.gae_lambda * lambda_return + (1 - self.gae_lambda) * value_t
                delta = self.gamma * jnp.where(truncated_t, value_t, (1.0 - done_t) * lambda_sum)
                lambda_return = soft_rewards[t] + delta
                return lambda_return, lambda_return

            init_lambda_return = next_values[-1]
            _, targets = jax.lax.scan(step, init_lambda_return, jnp.arange(self.nr_steps - 1, -1, -1))
            return targets[::-1]


        @jax.jit
        def update(policy_state, critic_state, batch_states, batch_actions, batch_rewards, batch_targets, batch_next_features, batch_terminations, batch_truncations, key):
            def critic_loss_fn(critic_params, observation_b, action_b, target_b, reward_b, next_features_b, terminated_b, truncated_b):
                _, critic_logits, pred_features, pred_reward = self.critic.apply(critic_params, observation_b, action_b)
                cdf_evals = jax.scipy.special.erf((self.hl_gauss_support - jnp.clip(target_b, self.v_min, self.v_max)[..., None]) / (jnp.sqrt(2) * self.hl_gauss_sigma))
                target_dist = (cdf_evals[..., 1:] - cdf_evals[..., :-1]) / (cdf_evals[..., -1:] - cdf_evals[..., :1])
                critic_update_loss = optax.softmax_cross_entropy(critic_logits, target_dist)
                auxiliary_loss = jnp.mean(
                    jnp.concatenate([(pred_features - next_features_b) ** 2, (pred_reward - reward_b[..., None]) ** 2], axis=-1), axis=-1,
                )
                critic_update_loss = (1.0 - truncated_b) * critic_update_loss
                auxiliary_loss = (1.0 - truncated_b) * (1.0 - terminated_b) * auxiliary_loss
                loss = jnp.mean(critic_update_loss) + self.auxiliary_loss_coefficient * jnp.mean(auxiliary_loss)
                value = jnp.sum(jax.nn.softmax(critic_logits, axis=-1) * self.hl_gauss_centers, axis=-1)
                metrics = {
                    "loss/critic_loss": critic_update_loss.mean(),
                    "loss/auxiliary_loss": auxiliary_loss.mean(),
                    "q/q_mean": value.mean(),
                    "q/explained_variance": 1 - jnp.var(target_b - value) / (jnp.var(target_b) + 1e-8),
                }
                return loss, metrics


            def policy_loss_fn(policy_params, critic_params, old_policy_params, observation_b, kl_key, sample_key):
                loc, log_std, log_entropy_coefficient, log_kl_coefficient = self.policy.apply(policy_params, observation_b)
                new_action, new_log_prob = self.policy.sample_and_log_prob(loc, log_std, sample_key)
                _, critic_logits, _, _ = self.critic.apply(critic_params, observation_b, new_action)
                value = jnp.sum(jax.nn.softmax(critic_logits, axis=-1) * self.hl_gauss_centers, axis=-1)

                old_loc, old_log_std, _, _ = self.policy.apply(old_policy_params, observation_b)
                broadcast_shape = (self.nr_kl_samples,) + old_loc.shape
                old_loc_b = jnp.broadcast_to(old_loc, broadcast_shape)
                old_log_std_b = jnp.broadcast_to(old_log_std, broadcast_shape)
                old_actions, old_log_probs = self.policy.sample_and_log_prob(old_loc_b, old_log_std_b, kl_key)
                loc_b = jnp.broadcast_to(loc, broadcast_shape)
                log_std_b = jnp.broadcast_to(log_std, broadcast_shape)
                new_log_probs_at_old = self.policy.log_prob(loc_b, log_std_b, old_actions)
                kl = jnp.mean(old_log_probs - new_log_probs_at_old, axis=0)

                entropy_coefficient = jnp.exp(log_entropy_coefficient).squeeze()
                kl_coefficient = jnp.exp(log_kl_coefficient).squeeze()
                clipped_loss = jnp.where(
                    kl < self.kl_bound,
                    new_log_prob * jax.lax.stop_gradient(entropy_coefficient) - value,
                    kl * jax.lax.stop_gradient(kl_coefficient),
                )
                entropy = -new_log_prob
                target_entropy_value = self.target_entropy + entropy
                entropy_coefficient_loss = entropy_coefficient * jax.lax.stop_gradient(target_entropy_value)
                kl_coefficient_loss = -kl_coefficient * jax.lax.stop_gradient(kl - self.kl_bound)
                loss = jnp.mean(clipped_loss) + jnp.mean(entropy_coefficient_loss) + jnp.mean(kl_coefficient_loss)
                metrics = {
                    "loss/policy_loss": clipped_loss.mean(),
                    "loss/entropy_coefficient_loss": entropy_coefficient_loss.mean(),
                    "loss/kl_coefficient_loss": kl_coefficient_loss.mean(),
                    "entropy/entropy": entropy.mean(),
                    "entropy/entropy_coefficient": entropy_coefficient,
                    "kl/kl_divergence": kl.mean(),
                    "kl/kl_coefficient": kl_coefficient,
                    "q/policy_q_mean": value.mean(),
                }
                return loss, metrics


            old_policy_params = policy_state.params
            critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
            policy_grad_fn = jax.value_and_grad(policy_loss_fn, has_aux=True)


            def epoch_iteration(epoch_carry, epoch_key):
                policy_state, critic_state = epoch_carry
                shuffle_key, mb_key = jax.random.split(epoch_key)
                indices = jax.random.permutation(shuffle_key, self.batch_size)
                indices = indices.reshape((self.nr_minibatches, self.minibatch_size))

                def minibatch_update(carry, batch_indices):
                    policy_state, critic_state, mb_key = carry
                    mb_key, sample_key, kl_key = jax.random.split(mb_key, 3)

                    obs_mb = batch_states[batch_indices]
                    action_mb = batch_actions[batch_indices]
                    reward_mb = batch_rewards[batch_indices]
                    target_mb = batch_targets[batch_indices]
                    next_features_mb = batch_next_features[batch_indices]
                    terminated_mb = batch_terminations[batch_indices]
                    truncated_mb = batch_truncations[batch_indices]

                    (_, critic_metrics), critic_grads = critic_grad_fn(
                        critic_state.params, obs_mb, action_mb, target_mb, reward_mb, next_features_mb, terminated_mb, truncated_mb
                    )
                    critic_state = critic_state.apply_gradients(grads=critic_grads)

                    (_, policy_metrics), policy_grads = policy_grad_fn(
                        policy_state.params, critic_state.params, old_policy_params, obs_mb, kl_key, sample_key
                    )
                    policy_state = policy_state.apply_gradients(grads=policy_grads)

                    metrics = {**critic_metrics, **policy_metrics}
                    metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grads)
                    metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grads)
                    return (policy_state, critic_state, mb_key), metrics

                (policy_state, critic_state, _), metrics = jax.lax.scan(minibatch_update, (policy_state, critic_state, mb_key), indices)
                return (policy_state, critic_state), metrics

            key, epoch_key = jax.random.split(key)
            epoch_keys = jax.random.split(epoch_key, self.nr_epochs)
            (policy_state, critic_state), metrics = jax.lax.scan(epoch_iteration, (policy_state, critic_state), epoch_keys)

            mean_metrics = tree.map_structure(lambda x: jnp.mean(x), metrics)
            mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            return policy_state, critic_state, mean_metrics, key


        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=np.float32),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=np.float32),
            rewards=np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32),
            soft_rewards=np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32),
            next_features=np.zeros((self.nr_steps, self.nr_envs, self.config.algorithm.critic_hidden_dim), dtype=np.float32),
            next_values=np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32),
            terminations=np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32),
            truncations=np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
            if logging_time_prev:
                time_metrics["time/logging_time_prev"] = logging_time_prev


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                state_jax = jnp.asarray(state, dtype=jnp.float32)
                if self.normalize_observation:
                    self.observation_normalizer_state = update_observation_normalizer(self.observation_normalizer_state, state_jax)
                normalized_state = preprocess_observation(self.observation_normalizer_state, state_jax)
                action, self.key = get_action(self.policy_state, normalized_state, self.key)
                action_np = jax.device_get(action)
                next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(self.get_processed_action(action)))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                        saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                normalized_actual_next_state = preprocess_observation(
                    self.observation_normalizer_state, jnp.asarray(actual_next_state, dtype=jnp.float32),
                )
                next_features, next_value, soft_reward, self.key = evaluate_next(
                    self.policy_state, self.critic_state, normalized_actual_next_state, reward, self.key
                )

                batch.states[step] = jax.device_get(normalized_state)
                batch.actions[step] = action_np
                batch.rewards[step] = reward
                batch.soft_rewards[step] = jax.device_get(soft_reward)
                batch.next_features[step] = jax.device_get(next_features)
                batch.next_values[step] = jax.device_get(next_value)
                batch.terminations[step] = terminated
                batch.truncations[step] = truncated
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating TD-lambda targets
            target_values = compute_td_lambda_targets(batch.soft_rewards, batch.next_values, batch.terminations, batch.truncations)

            calc_target_end_time = time.time()
            time_metrics["time/calc_target_time"] = calc_target_end_time - acting_end_time


            # Optimizing
            batch_states_flat = batch.states.reshape((-1,) + self.os_shape)
            batch_actions_flat = batch.actions.reshape((-1,) + self.as_shape)
            batch_rewards_flat = batch.rewards.reshape(-1)
            batch_targets_flat = jax.device_get(target_values).reshape(-1)
            batch_next_features_flat = batch.next_features.reshape((-1, batch.next_features.shape[-1]))
            batch_terminations_flat = batch.terminations.reshape(-1)
            batch_truncations_flat = batch.truncations.reshape(-1)

            self.policy_state, self.critic_state, optimization_metrics, self.key = update(
                self.policy_state, self.critic_state,
                batch_states_flat, batch_actions_flat, batch_rewards_flat, batch_targets_flat,
                batch_next_features_flat, batch_terminations_flat, batch_truncations_flat, self.key,
            )
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_target_end_time


            # Evaluating
            evaluation_metrics = {}
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    normalized_eval_state = preprocess_observation(
                        self.observation_normalizer_state, jnp.asarray(eval_state, dtype=jnp.float32),
                    )
                    eval_action = get_deterministic_action(self.policy_state, normalized_eval_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(self.get_processed_action(eval_action)))
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                self.set_train_mode()

            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time


            # Logging
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
                    if mean_value == mean_value:
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value

            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

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
        model = REPPO(config, train_env, eval_env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state,
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
        @jax.jit
        def get_action(policy_state, state):
            loc, _, _, _ = self.policy.apply(policy_state.params, state)
            return self.policy.deterministic_action(loc)

        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                state = observation_normalizer.normalize_observation(self.observation_normalizer_state, state) if self.normalize_observation else state
                action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(self.get_processed_action(action)))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
