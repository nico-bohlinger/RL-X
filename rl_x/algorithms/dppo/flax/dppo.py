import os
import shutil
import json
from functools import partial
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

from rl_x.algorithms.dppo.flax.general_properties import GeneralProperties
from rl_x.algorithms.dppo.flax.policy import get_policy
from rl_x.algorithms.dppo.flax.critic import get_critic
from rl_x.algorithms.dppo.flax import observation_normalizer
from rl_x.algorithms.dppo.flax import reward_normalizer

rlx_logger = logging.getLogger("rl_x")


class DPPO:
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
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clipping_epsilon = config.algorithm.clipping_epsilon
        self.clipping_epsilon_base = config.algorithm.clipping_epsilon_base
        self.clipping_epsilon_rate = config.algorithm.clipping_epsilon_rate
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.target_kl = config.algorithm.target_kl
        self.reward_scaling = config.algorithm.reward_scaling
        self.normalize_reward = config.algorithm.normalize_reward
        self.reward_clip = config.algorithm.reward_clip
        self.normalize_observation = config.algorithm.normalize_observation
        self.action_rescaling = config.algorithm.action_rescaling
        self.diffusion_steps = config.algorithm.diffusion_steps
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.policy_hidden_dims = tuple(config.algorithm.policy_hidden_dims)
        self.critic_hidden_dims = tuple(config.algorithm.critic_hidden_dims)
        self.denoising_std = config.algorithm.denoising_std
        self.denoising_discount = config.algorithm.denoising_discount
        self.denoised_clip_value = config.algorithm.denoised_clip_value
        self.noise_clip_value = config.algorithm.noise_clip_value
        self.log_probability_min = config.algorithm.log_probability_min
        self.log_probability_max = config.algorithm.log_probability_max
        self.advantage_quantile_min = config.algorithm.advantage_quantile_min
        self.advantage_quantile_max = config.algorithm.advantage_quantile_max
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.batch_size = self.nr_envs * self.nr_steps
        self.optimization_batch_size = self.batch_size * self.diffusion_steps
        self.nr_updates = self.total_timesteps // self.batch_size
        self.nr_minibatches = self.optimization_batch_size // self.minibatch_size
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape
        self.action_dimension = self.as_shape[0]
        self.action_low = jnp.asarray(self.train_env.single_action_space.low)
        self.action_high = jnp.asarray(self.train_env.single_action_space.high)
        cosine_positions = jnp.linspace(0.0, self.diffusion_steps + 1, self.diffusion_steps + 1)
        alpha_cumulative = jnp.cos((cosine_positions / (self.diffusion_steps + 1) + 0.008) / 1.008 * jnp.pi * 0.5) ** 2
        alpha_cumulative /= alpha_cumulative[0]
        self.betas = jnp.clip(1.0 - alpha_cumulative[1:] / alpha_cumulative[:-1], 0.0, 0.999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumulative = jnp.cumprod(self.alphas)
        self.alphas_cumulative_previous = jnp.concatenate([jnp.ones(1), self.alphas_cumulative[:-1]])
        self.sqrt_reciprocal_alphas_cumulative = jnp.sqrt(1.0 / self.alphas_cumulative)
        self.sqrt_reciprocal_minus_one_alphas_cumulative = jnp.sqrt(1.0 / self.alphas_cumulative - 1.0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumulative_previous) / (1.0 - self.alphas_cumulative)
        self.posterior_mean_coefficient_1 = self.betas * jnp.sqrt(self.alphas_cumulative_previous) / (1.0 - self.alphas_cumulative)
        self.posterior_mean_coefficient_2 = (1.0 - self.alphas_cumulative_previous) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumulative)

        if self.nr_updates == 0:
            raise ValueError("The total number of timesteps must contain at least one rollout batch.")
        if self.optimization_batch_size % self.minibatch_size != 0:
            raise ValueError("The denoising-MDP batch must be divisible by the minibatch size.")
        if self.diffusion_steps < 2:
            raise ValueError("DPPO requires at least two diffusion steps.")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be positive and divisible by two.")
        if self.denoising_std <= 0.0:
            raise ValueError("Denoising standard deviation must be positive.")
        if len(self.policy_hidden_dims) % 2 != 1 or len(self.critic_hidden_dims) % 2 != 1:
            raise ValueError("Residual networks require an odd number of hidden dimensions.")
        if not 0.0 <= self.advantage_quantile_min < self.advantage_quantile_max <= 1.0:
            raise ValueError("Advantage quantiles must be ordered inside [0, 1].")

        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)
        dummy_observation = jnp.asarray([self.train_env.single_observation_space.sample()])
        dummy_action = jnp.zeros(dummy_observation.shape[:-1] + self.as_shape)
        dummy_timestep = jnp.zeros(dummy_observation.shape[:-1] + (1,))

        self.policy = get_policy(config, self.train_env)
        self.critic = get_critic(config, self.train_env)

        def policy_linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.policy_learning_rate * fraction

        policy_learning_rate = policy_linear_schedule if self.anneal_learning_rate else self.policy_learning_rate
        if self.max_grad_norm != -1.0:
            policy_optimizer = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.inject_hyperparams(optax.adam)(learning_rate=policy_learning_rate))
        else:
            policy_optimizer = optax.chain(optax.inject_hyperparams(optax.adam)(learning_rate=policy_learning_rate))
        critic_optimizer = optax.chain(optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_learning_rate))
        self.policy_state = TrainState.create(apply_fn=self.policy.apply, params=self.policy.init(policy_key, dummy_observation, dummy_action, dummy_timestep), tx=policy_optimizer)
        self.critic_state = TrainState.create(apply_fn=self.critic.apply, params=self.critic.init(critic_key, dummy_observation), tx=critic_optimizer)
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)
        self.reward_normalizer_state = reward_normalizer.init_reward_normalizer_state(self.nr_envs)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def normalize(self, normalizer_state, observation):
        if self.normalize_observation:
            return observation_normalizer.normalize_observation(normalizer_state, observation)
        return observation


    def compute_transition_log_likelihood(self, policy_params, normalized_observation, path_current, path_next, denoising_index):
        if path_current.ndim == normalized_observation.ndim + 1:
            normalized_observation = jnp.broadcast_to(normalized_observation[..., None, :], path_current.shape[:-1] + (normalized_observation.shape[-1],))
        diffusion_timestep = self.diffusion_steps - denoising_index - 1
        timestep = diffusion_timestep[..., None].astype(jnp.float32)
        timestep = jnp.broadcast_to(timestep, path_current.shape[:-1] + (1,))
        predicted_noise = self.policy.apply(policy_params, normalized_observation, path_current, timestep)
        reconstructed_action = self.sqrt_reciprocal_alphas_cumulative[diffusion_timestep][..., None] * path_current - self.sqrt_reciprocal_minus_one_alphas_cumulative[diffusion_timestep][..., None] * predicted_noise
        reconstructed_action = jnp.clip(reconstructed_action, -self.denoised_clip_value, self.denoised_clip_value)
        transition_mean = self.posterior_mean_coefficient_1[diffusion_timestep][..., None] * reconstructed_action + self.posterior_mean_coefficient_2[diffusion_timestep][..., None] * path_current
        transition_std = jnp.maximum(jnp.sqrt(self.posterior_variance[diffusion_timestep])[..., None], self.denoising_std)
        standardized_noise = (path_next - transition_mean) / transition_std
        return -0.5 * standardized_noise ** 2 - 0.5 * jnp.log(2.0 * jnp.pi * transition_std ** 2)


    def sample_action(self, policy_params, normalizer_state, observation, key, deterministic=False):
        normalized_observation = self.normalize(normalizer_state, observation)
        key, initial_key, noise_key = jax.random.split(key, 3)
        initial_action = jax.random.normal(initial_key, observation.shape[:-1] + self.as_shape)
        noise_path = jax.random.normal(noise_key, (self.diffusion_steps,) + initial_action.shape)

        def denoising_step(noisy_action, inputs):
            diffusion_timestep, noise = inputs
            timestep = jnp.full(observation.shape[:-1] + (1,), diffusion_timestep, dtype=jnp.float32)
            predicted_noise = self.policy.apply(policy_params, normalized_observation, noisy_action, timestep)
            reconstructed_action = self.sqrt_reciprocal_alphas_cumulative[diffusion_timestep] * noisy_action - self.sqrt_reciprocal_minus_one_alphas_cumulative[diffusion_timestep] * predicted_noise
            reconstructed_action = jnp.clip(reconstructed_action, -self.denoised_clip_value, self.denoised_clip_value)
            transition_mean = self.posterior_mean_coefficient_1[diffusion_timestep] * reconstructed_action + self.posterior_mean_coefficient_2[diffusion_timestep] * noisy_action
            transition_std = jnp.maximum(jnp.sqrt(self.posterior_variance[diffusion_timestep]), 1e-3 if deterministic else self.denoising_std)
            transition_std = jnp.where(deterministic & (diffusion_timestep == 0), 0.0, transition_std)
            next_action = transition_mean + transition_std * jnp.clip(noise, -self.noise_clip_value, self.noise_clip_value)
            return next_action, noisy_action

        action, path = jax.lax.scan(denoising_step, initial_action, (jnp.arange(self.diffusion_steps - 1, -1, -1), noise_path))
        path = jnp.moveaxis(path, 0, -2)
        full_path = jnp.concatenate([path, action[..., None, :]], axis=-2)
        behavior_log_likelihood = self.compute_transition_log_likelihood(policy_params, normalized_observation, full_path[..., :-1, :], full_path[..., 1:, :], jnp.arange(self.diffusion_steps))
        if self.action_rescaling:
            processed_action = self.action_low + 0.5 * (action + 1.0) * (self.action_high - self.action_low)
        else:
            processed_action = action
        return key, action, processed_action, full_path, behavior_log_likelihood


    def train(self):
        @partial(jax.jit, static_argnames=("deterministic",))
        def get_action(policy_params, normalizer_state, observation, key, deterministic=False):
            return self.sample_action(policy_params, normalizer_state, observation, key, deterministic)


        @jax.jit
        def calculate_advantages(critic_state, next_states, rewards, values, terminations, truncations):
            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

            def advantage_step(next_advantage, inputs):
                reward, value, next_value, terminated, truncated = inputs
                delta = self.reward_scaling * reward + self.gamma * (1.0 - terminated) * next_value - value
                continuation = (1.0 - terminated) * (1.0 - truncated)
                advantage = delta + self.gamma * self.gae_lambda * continuation * next_advantage
                return advantage, advantage

            _, advantages = jax.lax.scan(advantage_step, jnp.zeros_like(values[-1]), (rewards, values, next_values, terminations, truncations), reverse=True)
            return advantages, advantages + values


        @jax.jit
        def update(policy_state, critic_state, states, actions, full_paths, behavior_log_likelihoods, advantages, returns, key):
            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_full_paths = full_paths.reshape((-1, self.diffusion_steps + 1) + self.as_shape)
            batch_behavior_log_likelihoods = behavior_log_likelihoods.reshape((-1, self.diffusion_steps, self.action_dimension))
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)

            def loss_fn(policy_params, critic_params, state_b, action_b, full_path_b, denoising_index_b, behavior_log_likelihood_b, advantage_b, return_b):
                current_log_likelihood = self.compute_transition_log_likelihood(policy_params, state_b, full_path_b[..., 0, :], full_path_b[..., 1, :], denoising_index_b)
                current_log_likelihood = jnp.clip(current_log_likelihood, self.log_probability_min, self.log_probability_max)
                behavior_log_likelihood_b = jnp.clip(behavior_log_likelihood_b, self.log_probability_min, self.log_probability_max)
                log_ratio = jnp.mean(current_log_likelihood - behavior_log_likelihood_b, axis=-1)
                ratio = jnp.exp(log_ratio)
                normalized_advantage = (advantage_b - jnp.mean(advantage_b)) / (jnp.std(advantage_b) + 1e-8)
                normalized_advantage = jnp.clip(normalized_advantage, jnp.quantile(normalized_advantage, self.advantage_quantile_min), jnp.quantile(normalized_advantage, self.advantage_quantile_max))
                normalized_advantage *= self.denoising_discount ** (self.diffusion_steps - denoising_index_b - 1)
                denoising_fraction = denoising_index_b / (self.diffusion_steps - 1)
                clipping_epsilon = self.clipping_epsilon_base + (self.clipping_epsilon - self.clipping_epsilon_base) * (jnp.exp(self.clipping_epsilon_rate * denoising_fraction) - 1.0) / (jnp.exp(self.clipping_epsilon_rate) - 1.0)
                surrogate = ratio * normalized_advantage
                clipped_surrogate = jnp.clip(ratio, 1.0 - clipping_epsilon, 1.0 + clipping_epsilon) * normalized_advantage
                policy_loss = -jnp.mean(jnp.minimum(surrogate, clipped_surrogate))
                value = self.critic.apply(critic_params, state_b).squeeze(-1)
                critic_loss = 0.5 * jnp.mean((value - return_b) ** 2)
                metrics = {
                    "loss/policy_gradient_loss": policy_loss,
                    "loss/critic_loss": critic_loss,
                    "policy_ratio/mean": jnp.mean(ratio),
                    "policy_ratio/min": jnp.min(ratio),
                    "policy_ratio/max": jnp.max(ratio),
                    "policy_ratio/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > clipping_epsilon),
                    "policy_ratio/approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
                    "policy_ratio/log_ratio_abs_max": jnp.max(jnp.abs(log_ratio)),
                    "diffusion/denoising_index_mean": jnp.mean(denoising_index_b),
                    "diffusion/clipping_epsilon_mean": jnp.mean(clipping_epsilon),
                    "policy/latent_action_abs_mean": jnp.mean(jnp.abs(action_b)),
                }
                return policy_loss + self.critic_coef * critic_loss, metrics

            grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
            key, shuffle_key = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.optimization_batch_size), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(shuffle_key, batch_indices, axis=1, independent=True).reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch_indices):
                policy_state, critic_state, update_active = carry
                transition_indices = minibatch_indices // self.diffusion_steps
                denoising_indices = minibatch_indices % self.diffusion_steps

                def perform_update(states):
                    policy_state, critic_state = states
                    (unused_loss, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(policy_state.params, critic_state.params, batch_states[transition_indices], batch_actions[transition_indices], jnp.stack([batch_full_paths[transition_indices, denoising_indices], batch_full_paths[transition_indices, denoising_indices + 1]], axis=-2), denoising_indices, batch_behavior_log_likelihoods[transition_indices, denoising_indices], batch_advantages[transition_indices], batch_returns[transition_indices])
                    policy_state = policy_state.apply_gradients(grads=policy_gradients)
                    critic_state = critic_state.apply_gradients(grads=critic_gradients)
                    metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                    metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                    metrics["optimization/update_active"] = jnp.ones(())
                    return (policy_state, critic_state), metrics

                def skip_update(states):
                    return states, {"loss/policy_gradient_loss": jnp.zeros(()), "loss/critic_loss": jnp.zeros(()), "policy_ratio/mean": jnp.ones(()), "policy_ratio/min": jnp.ones(()), "policy_ratio/max": jnp.ones(()), "policy_ratio/clip_fraction": jnp.zeros(()), "policy_ratio/approx_kl": jnp.zeros(()), "policy_ratio/log_ratio_abs_max": jnp.zeros(()), "diffusion/denoising_index_mean": jnp.zeros(()), "diffusion/clipping_epsilon_mean": jnp.zeros(()), "policy/latent_action_abs_mean": jnp.zeros(()), "gradients/policy_grad_norm": jnp.zeros(()), "gradients/critic_grad_norm": jnp.zeros(()), "optimization/update_active": jnp.zeros(())}

                (policy_state, critic_state), metrics = jax.lax.cond(update_active, perform_update, skip_update, (policy_state, critic_state))
                if self.target_kl is not None:
                    update_active &= metrics["policy_ratio/approx_kl"] <= self.target_kl
                return (policy_state, critic_state, update_active), metrics

            (policy_state, critic_state, unused_update_active), metrics = jax.lax.scan(minibatch_update, (policy_state, critic_state, jnp.ones((), dtype=jnp.bool_)), batch_indices)
            metrics = tree.map_structure(jnp.mean, metrics)
            metrics["lr/policy_learning_rate"] = policy_state.opt_state[-1].hyperparams["learning_rate"]
            metrics["lr/critic_learning_rate"] = critic_state.opt_state[-1].hyperparams["learning_rate"]
            return policy_state, critic_state, metrics, key


        state, unused_info = self.train_env.reset()
        global_step = 0
        completed_updates = 0
        while global_step < self.total_timesteps:
            start_time = time.time()
            states = np.zeros((self.nr_steps, self.nr_envs) + self.os_shape, dtype=np.float32)
            next_states = np.zeros_like(states)
            actions = np.zeros((self.nr_steps, self.nr_envs) + self.as_shape, dtype=np.float32)
            full_paths = np.zeros((self.nr_steps, self.nr_envs, self.diffusion_steps + 1) + self.as_shape, dtype=np.float32)
            behavior_log_likelihoods = np.zeros((self.nr_steps, self.nr_envs, self.diffusion_steps, self.action_dimension), dtype=np.float32)
            rewards = np.zeros((self.nr_steps, self.nr_envs), dtype=np.float32)
            values = np.zeros_like(rewards)
            terminations = np.zeros_like(rewards)
            truncations = np.zeros_like(rewards)
            step_info_collection = {}

            # Acting
            for step in range(self.nr_steps):
                if self.normalize_observation:
                    self.observation_normalizer_state = observation_normalizer.update_observation_normalizer(self.observation_normalizer_state, state)
                normalized_state = self.normalize(self.observation_normalizer_state, state)
                self.key, action, processed_action, full_path, behavior_log_likelihood = get_action(self.policy_state.params, self.observation_normalizer_state, state, self.key)
                value = self.critic.apply(self.critic_state.params, normalized_state).squeeze(-1)
                next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
                actual_next_state = next_state.copy()
                for index, done in enumerate(terminated | truncated):
                    if done:
                        actual_next_state[index] = np.asarray(self.train_env.get_final_observation_at_index(info, index))
                states[step] = normalized_state
                next_states[step] = self.normalize(self.observation_normalizer_state, actual_next_state)
                actions[step] = action
                full_paths[step] = full_path
                behavior_log_likelihoods[step] = behavior_log_likelihood
                rewards[step] = reward
                values[step] = value
                terminations[step] = terminated
                truncations[step] = truncated
                for name, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(name, []).extend(info_value)
                state = next_state
                global_step += self.nr_envs

            # Calculating advantages and returns
            normalized_rewards = rewards
            if self.normalize_reward:
                self.reward_normalizer_state, normalized_rewards = reward_normalizer.normalize_reward(self.reward_normalizer_state, rewards, terminations, truncations, self.gamma, self.reward_clip)
            advantages, returns = calculate_advantages(self.critic_state, next_states, normalized_rewards, values, terminations, truncations)
            # Optimizing
            self.policy_state, self.critic_state, metrics, self.key = update(self.policy_state, self.critic_state, states, actions, full_paths, behavior_log_likelihoods, advantages, returns, self.key)
            completed_updates += 1
            metrics["v_value/explained_variance"] = 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
            metrics["time/sps"] = self.batch_size / (time.time() - start_time)
            metrics["steps/nr_env_steps"] = global_step
            metrics["steps/nr_updates"] = completed_updates * self.nr_epochs * self.nr_minibatches
            for name, values_collection in step_info_collection.items():
                metric_group = "rollout" if name in ["episode_return", "episode_length"] else "env_info"
                metrics[f"{metric_group}/{name}"] = np.mean(values_collection)

            # Evaluating
            if self.evaluation_frequency != -1 and global_step % self.evaluation_frequency == 0:
                eval_state, unused_info = self.eval_env.reset()
                completed_episodes = 0
                while completed_episodes < self.evaluation_episodes:
                    self.key, unused_action, eval_action, unused_path, unused_log_likelihood = get_action(self.policy_state.params, self.observation_normalizer_state, eval_state, self.key, True)
                    eval_state, unused_reward, eval_terminated, eval_truncated, unused_info = self.eval_env.step(jax.device_get(eval_action))
                    completed_episodes += int(np.sum(eval_terminated | eval_truncated))

            # Saving
            if self.save_model and global_step >= self.total_timesteps:
                self.save(self.policy_state, self.critic_state, self.observation_normalizer_state, self.reward_normalizer_state)

            # Logging
            self.start_logging(global_step)
            for name, value in metrics.items():
                self.log(name, np.asarray(value), global_step)
            self.end_logging()


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


    def save(self, policy_state, critic_state, normalizer_state, reward_normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "observation_normalizer": normalizer_state,
            "reward_normalizer": reward_normalizer_state,
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
        model = DPPO(config, train_env, eval_env, run_path, writer)
        target = {
            "policy": model.policy_state,
            "critic": model.critic_state,
            "observation_normalizer": model.observation_normalizer_state,
            "reward_normalizer": model.reward_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(checkpoint_directory, item=target, restore_args=restore_args)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        model.reward_normalizer_state = checkpoint["reward_normalizer"]
        shutil.rmtree(checkpoint_directory)
        return model


    def test(self, episodes):
        state, unused_info = self.eval_env.reset()
        completed_episodes = 0
        while completed_episodes < episodes:
            self.key, unused_action, processed_action, unused_path, unused_log_likelihood = self.sample_action(self.policy_state.params, self.observation_normalizer_state, state, self.key, deterministic=True)
            state, unused_reward, terminated, truncated, unused_info = self.eval_env.step(jax.device_get(processed_action))
            completed_episodes += int(np.sum(terminated | truncated))


    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
