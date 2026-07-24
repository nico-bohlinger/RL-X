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

from rl_x.algorithms.fpo.flax_full_jit.networks import FlowPolicy, ValueCritic
from rl_x.algorithms.reppo.flax_full_jit import observation_normalizer

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
        self.flow_steps = config.algorithm.flow_steps
        self.timestep_embed_dim = config.algorithm.timestep_embed_dim
        self.policy_hidden_dims = tuple(config.algorithm.policy_hidden_dims)
        self.critic_hidden_dims = tuple(config.algorithm.critic_hidden_dims)
        self.policy_output_scale = config.algorithm.policy_output_scale
        self.output_mode = config.algorithm.output_mode
        self.nr_flow_samples_per_action = config.algorithm.nr_flow_samples_per_action
        self.average_losses_before_exp = config.algorithm.average_losses_before_exp
        self.discretize_t_for_training = config.algorithm.discretize_t_for_training
        self.feather_std = config.algorithm.feather_std
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
        self.policy_observation_indices = getattr(self.train_env, "policy_observation_indices", jnp.arange(self.os_shape[0]))
        self.critic_observation_indices = getattr(self.train_env, "critic_observation_indices", jnp.arange(self.os_shape[0]))
        self.action_low = jnp.asarray(self.train_env.single_action_space.low)
        self.action_high = jnp.asarray(self.train_env.single_action_space.high)
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
        if self.output_mode not in ["u", "u_but_supervise_as_eps"]:
            raise ValueError("Output mode must be u or u_but_supervise_as_eps.")
        if self.flow_steps < 1:
            raise ValueError("Flow steps must be positive.")
        if self.timestep_embed_dim < 2 or self.timestep_embed_dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be positive and divisible by two.")
        if self.nr_flow_samples_per_action < 1:
            raise ValueError("The number of flow samples per action must be positive.")
        rlx_logger.info(f"Using device: {jax.default_backend()}")

        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, reset_key = jax.random.split(self.key, 4)
        reset_key = jax.random.split(reset_key, 1)
        env_state = self.train_env.reset(reset_key, False)
        dummy_observation = env_state.next_observation
        dummy_action = jnp.zeros(dummy_observation.shape[:-1] + self.as_shape)
        dummy_timestep = jnp.zeros(dummy_observation.shape[:-1] + (1,))

        self.policy = FlowPolicy(
            self.action_dimension,
            self.timestep_embed_dim,
            self.policy_hidden_dims,
            self.policy_output_scale,
            self.policy_observation_indices,
        )
        self.critic = ValueCritic(self.critic_hidden_dims, self.critic_observation_indices)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate
        optimizer = lambda: optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
        )
        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dummy_observation, dummy_action, dummy_timestep),
            tx=optimizer(),
        )
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dummy_observation),
            tx=optimizer(),
        )
        self.observation_normalizer_state = observation_normalizer.init_observation_normalizer_state(self.os_shape)

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()


    def normalize(self, normalizer_state, observation):
        if self.normalize_observation:
            return observation_normalizer.normalize_observation(normalizer_state, observation)
        return observation


    def compute_cfm_loss(self, policy_params, normalized_observation, action, epsilon, timestep):
        sample_shape = action.shape[:-1] + (self.nr_flow_samples_per_action,)
        observation = jnp.broadcast_to(normalized_observation[..., None, :], sample_shape + (normalized_observation.shape[-1],))
        noisy_action = timestep * epsilon + (1.0 - timestep) * action[..., None, :]
        network_prediction = self.policy.apply(policy_params, observation, noisy_action, timestep)
        if self.output_mode == "u":
            target = epsilon - action[..., None, :]
            return jnp.mean((network_prediction - target) ** 2, axis=-1)
        action_prediction = noisy_action - timestep * network_prediction
        epsilon_prediction = action_prediction + network_prediction
        return jnp.mean((epsilon - epsilon_prediction) ** 2, axis=-1)


    def sample_action(self, policy_params, normalizer_state, observation, key, deterministic=False):
        normalized_observation = self.normalize(normalizer_state, observation)
        key, sample_key, loss_key, feather_key = jax.random.split(key, 4)
        initial_action = jax.random.normal(sample_key, observation.shape[:-1] + self.as_shape)

        def euler_step(noisy_action, inputs):
            current_timestep, next_timestep = inputs
            timestep = jnp.full(observation.shape[:-1] + (1,), current_timestep)
            velocity = self.policy.apply(policy_params, normalized_observation, noisy_action, timestep)
            next_action = noisy_action + (next_timestep - current_timestep) * velocity
            return next_action, None

        action, _ = jax.lax.scan(
            euler_step,
            initial_action,
            (self.schedule_current, self.schedule_next),
        )
        if not deterministic:
            action += self.feather_std * jax.random.normal(feather_key, action.shape)

        loss_shape = observation.shape[:-1] + (self.nr_flow_samples_per_action,)
        epsilon_key, timestep_key = jax.random.split(loss_key)
        epsilon = jax.random.normal(epsilon_key, loss_shape + self.as_shape)
        if self.discretize_t_for_training:
            timestep_indices = jax.random.randint(timestep_key, loss_shape, 0, self.flow_steps)
            timestep = self.schedule_current[timestep_indices][..., None]
        else:
            timestep = jax.random.uniform(timestep_key, loss_shape + (1,))
        initial_statistic = self.compute_cfm_loss(
            policy_params, normalized_observation, action, epsilon, timestep
        )
        action_info = (epsilon, timestep, initial_statistic)

        processed_action = self.action_low + 0.5 * (jnp.tanh(action) + 1.0) * (self.action_high - self.action_low)
        return key, action, processed_action, action_info


    def train(self):
        def jitable_train_function(key, parallel_seed_id):
            key, reset_key = jax.random.split(key)
            env_state = self.train_env.reset(jax.random.split(reset_key, self.nr_envs), False)
            policy_state = self.policy_state
            critic_state = self.critic_state
            normalizer_state = self.observation_normalizer_state

            def multi_iteration(carry, multi_iteration_step):
                policy_state, critic_state, normalizer_state, env_state, key = carry

                def learning_iteration(carry, learning_iteration_step):
                    policy_state, critic_state, normalizer_state, env_state, key = carry

                    def rollout_step(carry, _):
                        env_state, normalizer_state, key = carry
                        observation = env_state.next_observation
                        if self.normalize_observation:
                            normalizer_state = observation_normalizer.update_observation_normalizer(
                                normalizer_state, observation
                            )
                        key, action, processed_action, action_info = self.sample_action(
                            policy_state.params, normalizer_state, observation, key
                        )
                        normalized_observation = self.normalize(normalizer_state, observation)
                        value = self.critic.apply(critic_state.params, normalized_observation).squeeze(-1)
                        env_state = self.train_env.step(env_state, processed_action)
                        normalized_next_observation = self.normalize(
                            normalizer_state, env_state.actual_next_observation
                        )
                        transition = (
                            normalized_observation,
                            normalized_next_observation,
                            action,
                            action_info,
                            env_state.reward,
                            value,
                            env_state.terminated,
                            env_state.truncated,
                            env_state.info,
                        )
                        if self.render:
                            if self.render_callback_type == "debug_callback":
                                jax.debug.callback(self.train_env.render, env_state)
                            else:
                                env_state = jax.experimental.io_callback(self.train_env.render, env_state, env_state)
                        return (env_state, normalizer_state, key), transition

                    (env_state, normalizer_state, key), batch = jax.lax.scan(
                        rollout_step, (env_state, normalizer_state, key), None, self.nr_steps
                    )
                    states, next_states, actions, action_info, rewards, values, terminations, truncations, infos = batch
                    next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)

                    def advantage_step(next_advantage, inputs):
                        reward, value, next_value, terminated, truncated = inputs
                        delta = self.reward_scaling * reward + self.gamma * (1.0 - terminated) * next_value - value
                        continuation = (1.0 - terminated) * (1.0 - truncated)
                        advantage = delta + self.gamma * self.gae_lambda * continuation * next_advantage
                        return advantage, advantage

                    _, advantages = jax.lax.scan(
                        advantage_step,
                        jnp.zeros_like(values[-1]),
                        (rewards, values, next_values, terminations, truncations),
                        reverse=True,
                    )
                    returns = advantages + values

                    batch_states = states.reshape((-1,) + self.os_shape)
                    batch_actions = actions.reshape((-1,) + self.as_shape)
                    batch_advantages = advantages.reshape(-1)
                    batch_returns = returns.reshape(-1)
                    batch_epsilon = action_info[0].reshape(
                        (-1, self.nr_flow_samples_per_action) + self.as_shape
                    )
                    batch_timestep = action_info[1].reshape((-1, self.nr_flow_samples_per_action, 1))
                    batch_initial_statistic = action_info[2].reshape((-1, self.nr_flow_samples_per_action))

                    def loss_fn(policy_params, critic_params, state_b, action_b, advantage_b, return_b,
                                epsilon_b, timestep_b, initial_statistic_b):
                        normalized_advantage = (advantage_b - jnp.mean(advantage_b)) / (jnp.std(advantage_b) + 1e-8)
                        current_statistic = self.compute_cfm_loss(
                            policy_params, state_b, action_b, epsilon_b, timestep_b
                        )
                        if self.average_losses_before_exp:
                            ratio = jnp.exp(
                                jnp.mean(initial_statistic_b, axis=-1, keepdims=True)
                                - jnp.mean(current_statistic, axis=-1, keepdims=True)
                            )
                        else:
                            ratio = jnp.exp(jnp.clip(initial_statistic_b - current_statistic, -3.0, 3.0))

                        surrogate = ratio * normalized_advantage[..., None]
                        clipped_surrogate = jnp.clip(
                            ratio, 1.0 - self.clipping_epsilon, 1.0 + self.clipping_epsilon
                        ) * normalized_advantage[..., None]
                        policy_loss = -jnp.mean(jnp.minimum(surrogate, clipped_surrogate))
                        value = self.critic.apply(critic_params, state_b).squeeze(-1)
                        critic_loss = 0.5 * jnp.mean((value - return_b) ** 2)
                        total_loss = policy_loss + self.critic_coef * critic_loss
                        metrics = {
                            "loss/policy_gradient_loss": policy_loss,
                            "loss/critic_loss": critic_loss,
                            "policy_ratio/mean": jnp.mean(ratio),
                            "policy_ratio/min": jnp.min(ratio),
                            "policy_ratio/max": jnp.max(ratio),
                            "policy_ratio/clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > self.clipping_epsilon),
                            "policy/latent_action_abs_mean": jnp.mean(jnp.abs(action_b)),
                        }
                        return total_loss, metrics

                    grad_loss_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)
                    key, shuffle_key = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(
                        shuffle_key, batch_indices, axis=1, independent=True
                    ).reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry
                        (_, metrics), (policy_gradients, critic_gradients) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_advantages[minibatch_indices],
                            batch_returns[minibatch_indices],
                            batch_epsilon[minibatch_indices],
                            batch_timestep[minibatch_indices],
                            batch_initial_statistic[minibatch_indices],
                        )
                        policy_state = policy_state.apply_gradients(grads=policy_gradients)
                        critic_state = critic_state.apply_gradients(grads=critic_gradients)
                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                        return (policy_state, critic_state), metrics

                    (policy_state, critic_state), optimization_metrics = jax.lax.scan(
                        minibatch_update, (policy_state, critic_state), batch_indices
                    )
                    optimization_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["v_value/explained_variance"] = 1.0 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
                    combined_metrics = tree.map_structure(
                        jnp.mean, {**infos, **optimization_metrics}
                    )

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

                    jax.debug.callback(
                        callback,
                        (combined_metrics, learning_iteration_step, multi_iteration_step, parallel_seed_id),
                    )
                    return (policy_state, critic_state, normalizer_state, env_state, key), None

                carry, _ = jax.lax.scan(
                    learning_iteration,
                    (policy_state, critic_state, normalizer_state, env_state, key),
                    jnp.arange(self.nr_updates_per_multi_learning_iteration),
                )
                policy_state, critic_state, normalizer_state, env_state, key = carry

                if self.save_model:
                    jax.debug.callback(self.save, policy_state, critic_state, normalizer_state)

                return (policy_state, critic_state, normalizer_state, env_state, key), None

            jax.lax.scan(
                multi_iteration,
                (policy_state, critic_state, normalizer_state, env_state, key),
                jnp.arange(self.nr_multi_learning_and_eval_save_iterations),
            )

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


    def save(self, policy_state, critic_state, normalizer_state):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "observation_normalizer": normalizer_state,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as stream:
            json.dump(self.config.algorithm.to_dict(), stream)
        shutil.make_archive(
            f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp"
        )
        os.rename(
            f"{self.save_path}/{self.latest_model_file_name}.zip",
            f"{self.save_path}/{self.latest_model_file_name}",
        )
        shutil.rmtree(f"{self.save_path}/tmp")
        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    @staticmethod
    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        split_path = config.runner.load_model.split("/")
        checkpoint_directory = os.path.abspath("/".join(split_path[:-1]))
        checkpoint_file_name = split_path[-1]
        shutil.unpack_archive(
            f"{checkpoint_directory}/{checkpoint_file_name}", f"{checkpoint_directory}/tmp", "zip"
        )
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
            "observation_normalizer": model.observation_normalizer_state,
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpoint = orbax.checkpoint.PyTreeCheckpointer().restore(
            checkpoint_directory, item=target, restore_args=restore_args
        )
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]
        model.observation_normalizer_state = checkpoint["observation_normalizer"]
        shutil.rmtree(checkpoint_directory)
        return model


    def test(self, episodes):
        rlx_logger.info("Testing runs infinitely. The episodes parameter is ignored.")
        key, reset_key = jax.random.split(self.key)
        env_state = self.eval_env.reset(jax.random.split(reset_key, self.nr_envs), True)

        @jax.jit
        def rollout(env_state, key):
            def step(carry, _):
                env_state, key = carry
                key, _, processed_action, _ = self.sample_action(
                    self.policy_state.params,
                    self.observation_normalizer_state,
                    env_state.next_observation,
                    key,
                    deterministic=True,
                )
                env_state = self.eval_env.step(env_state, processed_action)
                return (env_state, key), None

            return jax.lax.scan(step, (env_state, key), None, self.horizon)[0]

        while True:
            env_state, key = rollout(env_state, key)
            self.eval_env.render(env_state)
