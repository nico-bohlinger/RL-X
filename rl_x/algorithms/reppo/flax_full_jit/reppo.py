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

from rl_x.algorithms.reppo.flax_full_jit.general_properties import GeneralProperties
from rl_x.algorithms.reppo.flax_full_jit.policy import get_policy
from rl_x.algorithms.reppo.flax_full_jit.critic import get_critic
from rl_x.algorithms.reppo.flax_full_jit import observation_normalizer

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
        self.nr_parallel_seeds = config.algorithm.nr_parallel_seeds
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.render = config.environment.render
        self.render_callback_type = getattr(config.environment, "render_callback_type", "io_callback")
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
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.minibatch_size = self.batch_size // self.nr_minibatches
        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if config.algorithm.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * (self.total_timesteps // self.batch_size)
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.action_dimension = self.train_env.single_action_space.shape[0]
        self.as_shape = (self.action_dimension,)
        self.os_shape = self.train_env.single_observation_space.shape
        self.horizon = self.train_env.horizon
        self.target_entropy = self.action_dimension * config.algorithm.target_entropy_multiplier
        bin_width = (self.v_max - self.v_min) / (self.nr_bins - 1)
        self.hl_gauss_centers = jnp.linspace(self.v_min, self.v_max, self.nr_bins, dtype=jnp.float32)
        self.hl_gauss_support = jnp.linspace(self.v_min - bin_width / 2, self.v_max + bin_width / 2, self.nr_bins + 1, dtype=jnp.float32)
        self.hl_gauss_sigma = bin_width * 0.75

        if self.batch_size % self.nr_minibatches != 0:
            raise ValueError("The rollout batch size must be divisible by the number of minibatches.")
        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of batch size")

        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet. This is mainly limited by not being able to log multiple wandb runs at the same time.")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)

        self.policy, self.get_processed_action = get_policy(self.config, self.train_env)
        self.critic = get_critic(self.config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        env_state = self.train_env.reset(reset_key, False)
        dummy_observation = env_state.next_observation
        dummy_action = jnp.zeros(dummy_observation.shape[:-1] + (self.action_dimension,), dtype=jnp.float32)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_observation),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dummy_observation, dummy_action),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)

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
            observation_normalizer_state = self.observation_normalizer_state

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, critic_state, observation_normalizer_state, env_state, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, critic_state, observation_normalizer_state, env_state, key = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, observation_normalizer_state, env_state, key = single_rollout_carry

                        key, act_key, next_act_key = jax.random.split(key, 3)
                        raw_observation = env_state.next_observation
                        if self.normalize_observation:
                            observation_normalizer_state = observation_normalizer.update_observation_normalizer(
                                observation_normalizer_state, raw_observation,
                            )
                            observation = observation_normalizer.normalize_observation(observation_normalizer_state, raw_observation)
                        else:
                            observation = raw_observation
                        loc, log_std, log_entropy_coefficient, _ = self.policy.apply(policy_state.params, observation)
                        action, _ = self.policy.sample_and_log_prob(loc, log_std, act_key)

                        env_state = self.train_env.step(env_state, self.get_processed_action(action))

                        actual_next_observation = env_state.actual_next_observation
                        if self.normalize_observation:
                            actual_next_observation = observation_normalizer.normalize_observation(
                                observation_normalizer_state, actual_next_observation,
                            )
                        next_loc, next_log_std, _, _ = self.policy.apply(policy_state.params, actual_next_observation)
                        next_action, next_log_prob = self.policy.sample_and_log_prob(next_loc, next_log_std, next_act_key)
                        next_features, next_critic_logits, _, _ = self.critic.apply(critic_state.params, actual_next_observation, next_action)
                        next_value = jnp.sum(jax.nn.softmax(next_critic_logits, axis=-1) * self.hl_gauss_centers, axis=-1)

                        entropy_coefficient = jnp.exp(log_entropy_coefficient).squeeze()
                        soft_reward = env_state.reward - self.gamma * next_log_prob * entropy_coefficient

                        transition = (
                            observation,
                            action,
                            env_state.reward,
                            soft_reward,
                            next_features,
                            next_value,
                            env_state.terminated,
                            env_state.truncated,
                            env_state.info,
                        )

                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                def render(env_state):
                                    return self.train_env.render(env_state)
                                env_state = jax.experimental.io_callback(render, env_state, env_state)

                        return (policy_state, critic_state, observation_normalizer_state, env_state, key), transition

                    single_rollout_carry, batch = jax.lax.scan(single_rollout, learning_iteration_carry, None, self.nr_steps)
                    policy_state, critic_state, observation_normalizer_state, env_state, key = single_rollout_carry
                    observations, actions, rewards, soft_rewards, next_features, next_values, terminations, truncations, infos = batch


                    # Calculating TD-lambda targets
                    def compute_td_lambda(carry, t):
                        lambda_return = carry
                        value_t = next_values[t]
                        done_t = terminations[t]
                        truncated_t = truncations[t]
                        lambda_sum = self.gae_lambda * lambda_return + (1 - self.gae_lambda) * value_t
                        delta = self.gamma * jnp.where(truncated_t, value_t, (1.0 - done_t) * lambda_sum)
                        lambda_return = soft_rewards[t] + delta
                        return lambda_return, lambda_return

                    init_lambda_return = next_values[-1]
                    _, target_values = jax.lax.scan(compute_td_lambda, init_lambda_return, jnp.arange(self.nr_steps - 1, -1, -1))
                    target_values = target_values[::-1]


                    # Snapshot old policy for KL constraint
                    old_policy_params = policy_state.params


                    # Optimizing
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


                    batch_observations = observations.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_rewards = rewards.reshape(-1)
                    batch_target_values = target_values.reshape(-1)
                    batch_next_features = next_features.reshape((-1, next_features.shape[-1]))
                    batch_terminations = terminations.reshape(-1).astype(jnp.float32)
                    batch_truncations = truncations.reshape(-1).astype(jnp.float32)

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

                            obs_mb = batch_observations[batch_indices]
                            action_mb = batch_actions[batch_indices]
                            reward_mb = batch_rewards[batch_indices]
                            target_mb = batch_target_values[batch_indices]
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
                    (policy_state, critic_state), optimization_metrics = jax.lax.scan(epoch_iteration, (policy_state, critic_state), epoch_keys)

                    optimization_metrics = tree.map_structure(lambda x: jnp.mean(x), optimization_metrics)
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]


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

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, parallel_seed_id))

                    return (policy_state, critic_state, observation_normalizer_state, env_state, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(
                    learning_iteration,
                    (policy_state, critic_state, observation_normalizer_state, env_state, subkey),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                policy_state, critic_state, observation_normalizer_state, env_state, key = learning_iteration_carry


                # Evaluating
                if self.evaluation_active:
                    def single_eval_rollout(single_eval_rollout_carry, _):
                        policy_state, eval_env_state = single_eval_rollout_carry

                        eval_observation = eval_env_state.next_observation
                        if self.normalize_observation:
                            eval_observation = observation_normalizer.normalize_observation(
                                observation_normalizer_state, eval_observation,
                            )
                        eval_loc, _, _, _ = self.policy.apply(policy_state.params, eval_observation)
                        eval_action = self.policy.deterministic_action(eval_loc)
                        eval_env_state = self.eval_env.step(eval_env_state, self.get_processed_action(eval_action))

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
                    def save_with_check(policy_state, critic_state, observation_normalizer_state):
                        self.save(policy_state, critic_state, observation_normalizer_state)
                    jax.debug.callback(save_with_check, policy_state, critic_state, observation_normalizer_state)


                return (policy_state, critic_state, observation_normalizer_state, env_state, key), None

            final_carry, _ = jax.lax.scan(
                multi_learning_and_eval_save_iteration,
                (policy_state, critic_state, observation_normalizer_state, env_state, key),
                jnp.arange(self.nr_multi_learning_and_eval_save_iterations),
            )
            policy_state, critic_state, observation_normalizer_state, _, _ = final_carry
            return policy_state, critic_state, observation_normalizer_state


        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        policy_state, critic_state, observation_normalizer_state = jax.block_until_ready(
            train_function(seed_keys, jnp.arange(self.nr_parallel_seeds))
        )
        self.policy_state = jax.tree.map(lambda x: x[0], policy_state)
        self.critic_state = jax.tree.map(lambda x: x[0], critic_state)
        self.observation_normalizer_state = jax.tree.map(lambda x: x[0], observation_normalizer_state)
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


    def save(self, policy_state, critic_state, observation_normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "observation_normalizer": observation_normalizer_state,
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
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")

        @jax.jit
        def rollout(env_state):
            observation = env_state.next_observation
            if self.normalize_observation:
                observation = observation_normalizer.normalize_observation(self.observation_normalizer_state, observation)
            loc, _, _, _ = self.policy.apply(self.policy_state.params, observation)
            action = self.policy.deterministic_action(loc)
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
