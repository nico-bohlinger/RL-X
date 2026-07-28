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

from rl_x.algorithms.fpo.flax_full_jit.policy import get_policy
from rl_x.algorithms.fpo.flax_full_jit.critic import get_critic
from rl_x.algorithms.fpo.flax_full_jit import observation_normalizer

rlx_logger = logging.getLogger("rl_x")


class FPO:
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
        self.weight_decay = config.algorithm.weight_decay
        self.adam_beta1 = config.algorithm.adam_beta1
        self.adam_beta2 = config.algorithm.adam_beta2
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clipping_epsilon = config.algorithm.clipping_epsilon
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.reward_scaling = config.algorithm.reward_scaling
        self.normalize_observation = config.algorithm.normalize_observation
        self.observation_normalizer_epsilon = config.algorithm.observation_normalizer_epsilon
        self.observation_normalizer_max_count = config.algorithm.observation_normalizer_max_count
        self.flow_steps = config.algorithm.flow_steps
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.actor_scale = config.algorithm.actor_scale
        self.action_clip = config.algorithm.action_clip
        self.nr_flow_samples_per_action = config.algorithm.nr_flow_samples_per_action
        self.timestep_inverse_cdf_beta = config.algorithm.timestep_inverse_cdf_beta
        self.action_perturb_std = config.algorithm.action_perturb_std
        self.cfm_loss_clamp = config.algorithm.cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages_max = config.algorithm.cfm_loss_clamp_negative_advantages_max
        self.cfm_difference_clamp_max = config.algorithm.cfm_difference_clamp_max
        self.trust_region_mode = config.algorithm.trust_region_mode
        self.advantage_clamp = config.algorithm.advantage_clamp
        self.ema_decay = config.algorithm.ema_decay
        self.ema_warmup_steps = config.algorithm.ema_warmup_steps
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency
        self.evaluation_active = config.algorithm.evaluation_active

        self.batch_size = self.nr_envs * self.nr_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]
        self.horizon = self.train_env.horizon
        self.schedule_current = jnp.linspace(1.0, 0.0, self.flow_steps + 1)[:-1]
        self.schedule_next = jnp.linspace(1.0, 0.0, self.flow_steps + 1)[1:]

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("The rollout batch size must be divisible by the minibatch size.")
        if self.evaluation_and_save_frequency % self.batch_size != 0:
            raise ValueError("Evaluation and save frequency must be a multiple of the rollout batch size.")
        if self.nr_parallel_seeds > 1:
            raise ValueError("Parallel seeds are not supported yet.")
        if self.flow_steps < 1:
            raise ValueError("Flow steps must be positive.")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be positive and divisible by two.")
        if self.nr_flow_samples_per_action < 1:
            raise ValueError("The number of flow samples per action must be positive.")
        if self.timestep_inverse_cdf_beta <= 0.0:
            raise ValueError("The timestep inverse-CDF beta must be positive.")
        if self.observation_normalizer_epsilon < 0.0 or self.observation_normalizer_max_count <= 0:
            raise ValueError("Observation normalizer epsilon and maximum count are invalid.")
        if self.actor_scale <= 0.0:
            raise ValueError("Actor scale must be positive.")
        if self.ema_decay < 0.0 or self.ema_decay >= 1.0 or self.ema_warmup_steps < 0:
            raise ValueError("EMA decay and warmup are invalid.")
        if self.trust_region_mode not in ["ppo", "spo", "aspo"]:
            raise ValueError("Trust-region mode must be ppo, spo or aspo.")
        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)
        env_state = self.train_env.reset(reset_key, False)
        dummy_observation = env_state.next_observation
        dummy_action = jnp.zeros(dummy_observation.shape[:-1] + self.as_shape)
        dummy_timestep = jnp.zeros(dummy_observation.shape[:-1] + (1,))

        self.policy = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        optimizer = optax.chain(optax.inject_hyperparams(optax.adamw)(learning_rate=learning_rate, b1=self.adam_beta1, b2=self.adam_beta2, weight_decay=self.weight_decay))
        self.policy_state = TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, dummy_observation, dummy_action, dummy_timestep), tx=optimizer)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=self.critic.init(critic_key, dummy_observation), tx=optimizer)
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)
        self.ema_policy_params = self.policy_state.params
        self.completed_updates = jnp.zeros((), dtype=jnp.int32)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def normalize(self, normalizer_state, observation):
        if self.normalize_observation:
            return observation_normalizer.normalize_observation(normalizer_state, observation, self.observation_normalizer_epsilon)
        return observation


    def compute_cfm_loss(self, policy_params, normalized_observation, action, epsilon, timestep):
        sample_shape = action.shape[:-1] + (self.nr_flow_samples_per_action,)
        observation = jnp.broadcast_to(normalized_observation[..., None, :], sample_shape + (normalized_observation.shape[-1],))
        scaled_action = action / self.actor_scale
        noisy_action = timestep * epsilon + (1.0 - timestep) * scaled_action[..., None, :]
        network_prediction = self.policy.apply(policy_params, observation, noisy_action, timestep)
        target = epsilon - scaled_action[..., None, :]
        return jnp.sum((network_prediction - target) ** 2, axis=-1) / jnp.sqrt(self.action_dimension)


    def sample_action(self, policy_params, normalizer_state, observation, key, deterministic=False):
        normalized_observation = self.normalize(normalizer_state, observation)
        key, sample_key, loss_key, perturb_key = jax.random.split(key, 4)
        initial_action = jax.random.normal(sample_key, observation.shape[:-1] + self.as_shape)

        def euler_step(noisy_action, inputs):
            current_timestep, next_timestep = inputs
            timestep = jnp.full(observation.shape[:-1] + (1,), current_timestep)
            velocity = self.policy.apply(policy_params, normalized_observation, noisy_action, timestep)
            next_action = noisy_action + (next_timestep - current_timestep) * velocity
            return next_action, None

        action, _ = jax.lax.scan(euler_step, initial_action, (self.schedule_current, self.schedule_next))
        action *= self.actor_scale
        if not deterministic:
            action += self.action_perturb_std * jax.random.normal(perturb_key, action.shape)

        loss_shape = observation.shape[:-1] + (self.nr_flow_samples_per_action,)
        epsilon_key, timestep_key = jax.random.split(loss_key)
        epsilon = jax.random.normal(epsilon_key, loss_shape + self.as_shape)
        uniform_timestep = jax.random.uniform(timestep_key, loss_shape + (1,))
        timestep = 0.005 + 0.99 * (1.0 - (1.0 - uniform_timestep) ** (1.0 / self.timestep_inverse_cdf_beta))
        initial_statistic = self.compute_cfm_loss(policy_params, normalized_observation, action, epsilon, timestep)
        action_info = (epsilon, timestep, initial_statistic)

        processed_action = jnp.clip(action, -self.action_clip, self.action_clip)
        return key, action, processed_action, action_info


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key)
            env_state = self.train_env.reset(jax.random.split(reset_key, self.nr_envs), False)
            policy_state = self.policy_state
            critic_state = self.critic_state
            ema_policy_params = self.ema_policy_params
            normalizer_state = self.observation_normalizer_state

            def multi_iteration(carry, multi_iteration_step):
                policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key = carry

                def learning_iteration(carry, learning_iteration_step):
                    policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key = carry

                    # Acting
                    def rollout_step(carry, _):
                        env_state, normalizer_state, key = carry
                        observation = env_state.next_observation
                        if self.normalize_observation:
                            normalizer_state = observation_normalizer.update_observation_normalizer(normalizer_state, observation, self.observation_normalizer_max_count)
                        key, action, processed_action, action_info = self.sample_action(policy_state.params, normalizer_state, observation, key)
                        normalized_observation = self.normalize(normalizer_state, observation)
                        value = self.critic.apply(critic_state.params, normalized_observation).squeeze(-1)
                        env_state = self.train_env.step(env_state, processed_action)
                        normalized_next_observation = self.normalize(normalizer_state, env_state.actual_next_observation)
                        transition = normalized_observation, normalized_next_observation, action, action_info, env_state.reward, value, env_state.terminated, env_state.truncated, env_state.info
                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                env_state = jax.experimental.io_callback(self.train_env.render, env_state, env_state)
                        return (env_state, normalizer_state, key), transition

                    (env_state, normalizer_state, key), batch = jax.lax.scan(rollout_step, (env_state, normalizer_state, key), None, self.nr_steps)
                    states, next_states, actions, action_info, rewards, values, terminations, truncations, infos = batch
                    next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

                    # Calculating advantages and returns
                    def advantage_step(next_advantage, inputs):
                        reward, value, next_value, terminated, truncated = inputs
                        delta = self.reward_scaling * reward + self.gamma * (1.0 - terminated) * next_value - value
                        continuation = (1.0 - terminated) * (1.0 - truncated)
                        advantage = delta + self.gamma * self.gae_lambda * continuation * next_advantage
                        return advantage, advantage

                    _, advantages = jax.lax.scan(advantage_step, jnp.zeros_like(values[-1]), (rewards, values, next_values, terminations, truncations), reverse=True)
                    returns = advantages + values

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_epsilon = action_info[0].reshape((-1, self.nr_flow_samples_per_action) + self.as_shape)
                    batch_timestep = action_info[1].reshape((-1, self.nr_flow_samples_per_action, 1))
                    batch_initial_statistic = action_info[2].reshape((-1, self.nr_flow_samples_per_action))
                    batch_advantages = (batch_advantages - jnp.mean(batch_advantages)) / (jnp.std(batch_advantages) + 1e-8)
                    batch_advantages = jnp.clip(batch_advantages, -self.advantage_clamp, self.advantage_clamp)

                    # Optimizing
                    def loss_fn(policy_params, critic_params, state_b, action_b, advantage_b, return_b, epsilon_b, timestep_b, initial_statistic_b):
                        current_statistic = self.compute_cfm_loss(policy_params, state_b, action_b, epsilon_b, timestep_b)
                        initial_statistic_b = jnp.minimum(initial_statistic_b, self.cfm_loss_clamp)
                        current_statistic = jnp.minimum(current_statistic, self.cfm_loss_clamp)
                        current_statistic = jnp.where(advantage_b[..., None] < 0.0, jnp.minimum(current_statistic, self.cfm_loss_clamp_negative_advantages_max), current_statistic)
                        unclamped_log_ratio = initial_statistic_b - current_statistic
                        clamped_log_ratio = jnp.minimum(unclamped_log_ratio, self.cfm_difference_clamp_max)
                        log_ratio = unclamped_log_ratio + jax.lax.stop_gradient(clamped_log_ratio - unclamped_log_ratio)
                        ratio = jnp.exp(log_ratio)
                        surrogate = -advantage_b[..., None] * ratio
                        clipped_surrogate = -advantage_b[..., None] * jnp.clip(ratio, 1.0 - self.clipping_epsilon, 1.0 + self.clipping_epsilon)
                        ppo_loss = jnp.maximum(surrogate, clipped_surrogate)
                        spo_loss = -(ratio * advantage_b[..., None] - jnp.abs(advantage_b[..., None]) * (ratio - 1.0) ** 2 / (2.0 * self.clipping_epsilon))
                        if self.trust_region_mode == "ppo":
                            policy_loss = jnp.mean(ppo_loss)
                        elif self.trust_region_mode == "spo":
                            policy_loss = jnp.mean(spo_loss)
                        else:
                            policy_loss = jnp.mean(jnp.where(advantage_b[..., None] > 0.0, ppo_loss, spo_loss))
                        value = self.critic.apply(critic_params, state_b).squeeze(-1)
                        critic_loss = jnp.mean((value - return_b) ** 2)
                        total_loss = policy_loss + self.critic_coef * critic_loss
                        metrics = {
                            "loss/policy_gradient_loss": policy_loss,
                            "loss/critic_loss": critic_loss,
                            "policy_ratio/mean": jnp.mean(ratio),
                            "policy_ratio/min": jnp.min(ratio),
                            "policy_ratio/max": jnp.max(ratio),
                            "policy_ratio/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > self.clipping_epsilon),
                            "policy_ratio/log_ratio_unclamped_max": jnp.max(unclamped_log_ratio),
                            "policy_ratio/nonfinite_fraction": jnp.mean(~jnp.isfinite(ratio)),
                            "policy/latent_action_abs_mean": jnp.mean(jnp.abs(action_b)),
                            "cfm/initial_loss_mean": jnp.mean(initial_statistic_b),
                            "cfm/current_loss_mean": jnp.mean(current_statistic),
                        }
                        return total_loss, metrics

                    grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                    key, shuffle_key = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(shuffle_key, batch_indices, axis=1, independent=True).reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry
                        (_, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(policy_state.params, critic_state.params, batch_states[minibatch_indices], batch_actions[minibatch_indices], batch_advantages[minibatch_indices], batch_returns[minibatch_indices], batch_epsilon[minibatch_indices], batch_timestep[minibatch_indices], batch_initial_statistic[minibatch_indices])
                        combined_gradient_norm = jnp.sqrt(optax.global_norm(policy_gradients) ** 2 + optax.global_norm(critic_gradients) ** 2)
                        gradient_scale = jnp.minimum(1.0, self.max_grad_norm / (combined_gradient_norm + 1e-6))
                        policy_state = policy_state.apply_gradients(grads=tree.map_structure(lambda gradient: gradient * gradient_scale, policy_gradients))
                        critic_state = critic_state.apply_gradients(grads=tree.map_structure(lambda gradient: gradient * gradient_scale, critic_gradients))
                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        return (policy_state, critic_state), metrics

                    (policy_state, critic_state), optimization_metrics = jax.lax.scan(minibatch_update, (policy_state, critic_state), batch_indices)
                    combined_step = multi_iteration_step * self.nr_updates_per_multi_learning_iteration + learning_iteration_step + 1
                    if self.ema_decay > 0.0:
                        ema_policy_params = jax.tree.map(lambda ema_parameter, policy_parameter: jnp.where(combined_step == self.ema_warmup_steps, policy_parameter, jnp.where(combined_step > self.ema_warmup_steps, self.ema_decay * ema_parameter + (1.0 - self.ema_decay) * policy_parameter, ema_parameter)), ema_policy_params, policy_state.params)
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[0].hyperparams["learning_rate"]
                    optimization_metrics["policy/ema_active"] = combined_step > self.ema_warmup_steps
                    optimization_metrics["v_value/explained_variance"] = 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    combined_metrics = tree.map_structure(jnp.mean, {**infos, **optimization_metrics})

                    # Logging
                    def callback(callback_carry):
                        metrics, learning_iteration_step, multi_iteration_step, parallel_seed_id = callback_carry
                        current_time = time.time()
                        metrics["time/sps"] = int(self.batch_size / (current_time - self.last_time[parallel_seed_id]))
                        self.last_time[parallel_seed_id] = current_time
                        combined_step = multi_iteration_step * self.nr_updates_per_multi_learning_iteration + learning_iteration_step + 1
                        global_step = int(combined_step.item() * self.batch_size)
                        metrics["steps/nr_env_steps"] = global_step
                        metrics["steps/nr_updates"] = combined_step.item() * self.nr_epochs * self.nr_minibatches
                        self.start_logging(global_step)
                        for name, value in metrics.items():
                            self.log(name, np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, multi_iteration_step, parallel_seed_id))
                    return (policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key), None

                carry, _ = jax.lax.scan(learning_iteration, (policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key = carry
                completed_updates = (multi_iteration_step + 1) * self.nr_updates_per_multi_learning_iteration

                # Evaluating
                if self.evaluation_active:
                    def eval_rollout(eval_carry, _):
                        eval_env_state, key = eval_carry
                        policy_params = jax.tree.map(lambda ema_parameter, policy_parameter: jnp.where(completed_updates > self.ema_warmup_steps, ema_parameter, policy_parameter), ema_policy_params, policy_state.params)
                        key, unused_action, processed_action, unused_action_info = self.sample_action(policy_params, normalizer_state, eval_env_state.next_observation, key, True)
                        eval_env_state = self.eval_env.step(eval_env_state, processed_action)
                        return (eval_env_state, key), None

                    key, reset_key = jax.random.split(key)
                    eval_env_state = self.eval_env.reset(jax.random.split(reset_key, self.nr_envs), True)
                    (eval_env_state, key), _ = jax.lax.scan(eval_rollout, (eval_env_state, key), None, self.horizon)
                    evaluation_metrics = {
                        "eval/episode_return": jnp.mean(eval_env_state.info["rollout/episode_return"]),
                        "eval/episode_length": jnp.mean(eval_env_state.info["rollout/episode_length"]),
                    }

                    def callback(metrics_and_step):
                        metrics, combined_step = metrics_and_step
                        global_step = int(combined_step.item() * self.batch_size)
                        self.start_logging(global_step)
                        for name, value in metrics.items():
                            self.log(name, np.asarray(value), global_step)
                        self.end_logging()

                    jax.debug.callback(callback, (evaluation_metrics, completed_updates))

                # Saving
                if self.save_model:
                    jax.debug.callback(self.save, policy_state, critic_state, ema_policy_params, normalizer_state, completed_updates)

                return (policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key), None

            jax.lax.scan(multi_iteration, (policy_state, critic_state, ema_policy_params, normalizer_state, env_state, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))

        self.key, subkey = jax.random.split(self.key)
        seed_keys = jax.random.split(subkey, self.nr_parallel_seeds)
        train_function = jax.jit(jax.vmap(jitable_train_function))
        self.last_time = [time.time() for _ in range(self.nr_parallel_seeds)]
        self.start_time = deepcopy(self.last_time)
        jax.block_until_ready(train_function(seed_keys, jnp.arange(self.nr_parallel_seeds)))
        rlx_logger.info(f"Average time: {max(time.time() - start_time for start_time in self.start_time):.2f} s")


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            value = np.format_float_positional(value, trim="-")
            rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, policy_state, critic_state, ema_policy_params, normalizer_state, completed_updates):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "ema_policy": ema_policy_params,
            "observation_normalizer": normalizer_state,
            "completed_updates": completed_updates,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as stream:
            json.dump(self.config.algorithm.to_dict(), stream)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")
        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        split_path = config.runner.load_model.split("/")
        checkpoint_directory = os.path.abspath("/".join(split_path[:-1]))
        checkpoint_file_name = split_path[-1]
        shutil.unpack_archive(f"{checkpoint_directory}/{checkpoint_file_name}", f"{checkpoint_directory}/tmp", "zip")
        checkpoint_directory = f"{checkpoint_directory}/tmp"
        with open(f"{checkpoint_directory}/config_algorithm.json") as stream:
            loaded_algorithm_config = json.load(stream)
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = FPO(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "ema_policy": model.ema_policy_params,
            "observation_normalizer": model.observation_normalizer_state,
            "completed_updates": model.completed_updates,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_directory, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.ema_policy_params = checkpoint["ema_policy"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        model.completed_updates = checkpoint["completed_updates"]
        shutil.rmtree(checkpoint_directory)
        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")
        key, reset_key = jax.random.split(self.key)
        env_state = self.eval_env.reset(jax.random.split(reset_key, self.nr_envs), True)
        policy_params = self.ema_policy_params if self.ema_decay > 0.0 and int(self.completed_updates) > self.ema_warmup_steps else self.policy_state.params

        @jax.jit
        def rollout(env_state, key):
            def step(carry, _):
                env_state, key = carry
                key, _, processed_action, _ = self.sample_action(policy_params, self.observation_normalizer_state, env_state.next_observation, key, deterministic=True)
                env_state = self.eval_env.step(env_state, processed_action)
                return (env_state, key), None

            return jax.lax.scan(step, (env_state, key), None, self.horizon)[0]

        while True:
            env_state, key = rollout(env_state, key)
            self.eval_env.render(env_state)
